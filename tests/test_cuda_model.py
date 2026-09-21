"""Same-flat-batch regression gates, not cross-shape HF quality certification."""

from __future__ import annotations

import os

import pytest
import torch
from transformers import Qwen3Config as HFConfig

from tinyvllm.cache import BlockManager
from tinyvllm.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from tinyvllm.request import WorkClass
from tinyvllm.runtime.runner import ModelRunner
from paged_helpers import logical_kv, prepare, scenario


pytestmark = pytest.mark.skipif(
    os.environ.get("TINYVLLM_RUN_CUDA") != "1",
    reason="set TINYVLLM_RUN_CUDA=1 to run CUDA integration gates",
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _runners(model, *, block_size, num_blocks, max_model_len):
    runners = tuple(
        ModelRunner(
            model,
            block_size=block_size,
            num_blocks=num_blocks,
            max_model_len=max_model_len,
            attention_backend=backend,
        )
        for backend in ("reference", "cuda")
    )
    for runner in runners:
        runner.kv_cache.key.fill_(float("nan"))
        runner.kv_cache.value.fill_(float("nan"))
    return runners


def _run_pair(
    model,
    runners,
    manager,
    steps,
    monkeypatch,
    *,
    max_model_len,
    release_after,
    reused_request,
    check_tensors,
    check_logits,
):
    """Both runners consume identical teacher-forced rows and keep their own KV."""
    reference, cuda = runners
    assert reference.kv_cache.key.data_ptr() != cuda.kv_cache.key.data_ptr()
    read_sizes = []
    original_read = cuda.kv_cache.read

    def recorded_read(layer_idx, slots):
        read_sizes.append((layer_idx, slots.numel()))
        return original_read(layer_idx, slots)

    monkeypatch.setattr(cuda.kv_cache, "read", recorded_read)
    hidden = []
    forward_rows = []

    def record_forward(module, args, output):
        forward_rows.append(args[0].numel())
        hidden.append(output.detach().clone())

    hook = model.register_forward_hook(record_forward)
    active_lengths = {}
    retired_blocks = set()
    observed_reuse = False
    try:
        with torch.inference_mode():
            for step_id, records in enumerate(steps):
                transaction, batch = prepare(
                    manager,
                    step_id,
                    records,
                    max_model_len=max_model_len,
                )
                transaction.commit_before_dispatch()
                hidden.clear()
                forward_rows.clear()
                read_sizes.clear()

                expected = reference.run(batch)
                actual = cuda.run(batch)

                # Each backend runs the entire flat model once, even in mixed steps.
                assert forward_rows == [batch.num_query_tokens] * 2
                assert len(hidden) == 2
                assert torch.isfinite(hidden[0]).all()
                assert torch.isfinite(hidden[1]).all()
                if check_tensors:
                    torch.testing.assert_close(
                        hidden[1], hidden[0], atol=0.04, rtol=0.04,
                    )

                expected_reads = [
                    (layer, length)
                    for layer in range(model.config.num_hidden_layers)
                    for before, query_len, length in zip(
                        batch.context_lens_before,
                        batch.query_lens, batch.context_lens_after,
                    )
                    if before > 0 and query_len > 1
                ]
                assert read_sizes == expected_reads

                if expected is None:
                    assert actual is None
                    assert batch.num_samples == 0
                else:
                    assert actual is not None
                    assert actual.shape == expected.shape == (
                        batch.num_samples, model.config.vocab_size,
                    )
                    check_logits(actual, expected)

                active_lengths.update(zip(batch.request_ids, batch.context_lens_after))
                for request_id, length in active_lengths.items():
                    expected_kv = logical_kv(reference, manager, request_id, length)
                    actual_kv = logical_kv(cuda, manager, request_id, length)
                    for actual_tensor, expected_tensor in zip(actual_kv, expected_kv):
                        assert torch.isfinite(actual_tensor).all()
                        assert torch.isfinite(expected_tensor).all()
                        if check_tensors:
                            torch.testing.assert_close(
                                actual_tensor,
                                expected_tensor,
                                atol=0.04,
                                rtol=0.04,
                            )

                transaction.complete_writes()
                if step_id == release_after[0]:
                    request_id = release_after[1]
                    retired_blocks = {
                        ref.block_id for ref in manager.block_table(request_id)
                    }
                    manager.release_request(request_id)
                    del active_lengths[request_id]

                if reused_request in batch.request_ids:
                    fresh_blocks = {
                        ref.block_id for ref in manager.block_table(reused_request)
                    }
                    assert fresh_blocks <= retired_blocks
                    observed_reuse = True
    finally:
        hook.remove()
    assert observed_reuse


def test_small_bf16_model_preserves_mixed_history_and_uses_one_flat_forward(monkeypatch):
    _require_cuda()
    torch.manual_seed(773)
    config = HFConfig(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(
        Qwen3Config.from_hf(config), device="cuda", dtype=torch.bfloat16,
    ).eval()
    manager = BlockManager(block_size=2, num_blocks=8)
    runners = _runners(model, block_size=2, num_blocks=8, max_model_len=32)
    prefill, decode = WorkClass.PREFILL, WorkClass.DECODE
    steps = (*scenario(), (
        (44, (21, 22, 23), 3, True, prefill),
        (55, (24,), 1, True, prefill),
    ), (
        (55, (24, 25), 1, True, decode),
        (44, (21, 22, 23, 26), 1, True, decode),
    ))

    def check_logits(actual, expected):
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)

    _run_pair(
        model, runners, manager, steps, monkeypatch,
        max_model_len=32,
        release_after=(3, 22),
        reused_request=44,
        check_tensors=True,
        check_logits=check_logits,
    )


class _RealLogitBudget:
    """Fixed module regression budgets; never widen these after a failed run."""

    def __init__(self):
        self.mean_abs = 0.0
        self.max_abs = 0.0
        self.total_variation = 0.0
        self.rows = 0
        self.greedy_matches = 0
        self.stable_rows = 0

    def __call__(self, actual, expected):
        actual, expected = actual.float(), expected.float()
        assert torch.isfinite(actual).all()
        assert torch.isfinite(expected).all()
        errors = (actual - expected).abs()
        row_mean = errors.mean(dim=-1)
        row_max = errors.amax(dim=-1)
        total_variation = (
            actual.softmax(dim=-1) - expected.softmax(dim=-1)
        ).abs().sum(dim=-1) * 0.5
        matches = actual.argmax(dim=-1) == expected.argmax(dim=-1)
        top_two = expected.topk(2, dim=-1).values
        stable = top_two[:, 0] - top_two[:, 1] > 2 * row_max

        self.mean_abs = max(self.mean_abs, row_mean.max().item())
        self.max_abs = max(self.max_abs, row_max.max().item())
        self.total_variation = max(
            self.total_variation, total_variation.max().item(),
        )
        self.rows += expected.shape[0]
        self.greedy_matches += matches.sum().item()
        self.stable_rows += stable.sum().item()

        assert self.mean_abs <= 0.06, self.summary()
        assert self.max_abs <= 0.6, self.summary()
        assert self.total_variation <= 0.02, self.summary()
        assert matches[stable].all(), self.summary()

    def summary(self):
        return (
            f"same-flat-batch logits: max row mean_abs={self.mean_abs:.6f}, "
            f"max_abs={self.max_abs:.6f}, "
            f"max total_variation={self.total_variation:.6f}, "
            f"greedy matches={self.greedy_matches}/{self.rows}, "
            f"stable-margin rows={self.stable_rows}"
        )


def _long_scenario():
    prefill, decode = WorkClass.PREFILL, WorkClass.DECODE
    first = tuple(range(1, 66))
    second = tuple(range(200, 328))
    fresh = tuple(range(500, 565))
    return (
        ((10, first, 64, False, prefill), (22, second, 64, False, prefill)),
        ((10, first, 1, True, prefill), (22, second, 64, True, prefill)),
        (
            (22, second + (400,), 1, True, decode),
            (10, first + (401,), 1, True, decode),
        ),
        (
            (10, first + (401, 402), 1, True, decode),
            (22, second + (400, 403), 1, True, decode),
        ),
        (
            (10, first + (401, 402, 404), 1, True, decode),
            (44, fresh, 64, False, prefill),
        ),
        (
            (10, first + (401, 402, 404, 405), 1, True, decode),
            (44, fresh, 1, True, prefill),
        ),
        (
            (44, fresh + (406,), 1, True, decode),
            (10, first + (401, 402, 404, 405, 407), 1, True, decode),
        ),
    )


@pytest.mark.skipif(
    not os.environ.get("TINYVLLM_MODEL"),
    reason="set TINYVLLM_MODEL to local Qwen3 weights for the real model gate",
)
def test_real_qwen3_logits_stay_within_fixed_same_shape_budget(monkeypatch):
    _require_cuda()
    model = Qwen3ForCausalLM.from_pretrained(
        os.environ["TINYVLLM_MODEL"], device="cuda", dtype=torch.bfloat16,
    )
    manager = BlockManager(block_size=16, num_blocks=14)
    runners = _runners(model, block_size=16, num_blocks=14, max_model_len=256)
    budget = _RealLogitBudget()
    try:
        _run_pair(
            model, runners, manager, _long_scenario(), monkeypatch,
            max_model_len=256,
            release_after=(3, 22),
            reused_request=44,
            check_tensors=False,
            check_logits=budget,
        )
    finally:
        print(budget.summary())



@pytest.mark.skipif(
    not os.environ.get("TINYVLLM_MODEL"),
    reason="set TINYVLLM_MODEL for the extended chunk regression",
)
def test_real_qwen3_long_chunks_against_previous_cuda_attention():
    """Isolate the chunk backend change; both runners keep identical decode."""
    from tinyvllm.runtime.attention import CudaAttention, ReferenceAttention

    class PreviousCudaAttention(CudaAttention):
        _make_mask = ReferenceAttention._make_mask

    _require_cuda()
    model = Qwen3ForCausalLM.from_pretrained(
        os.environ["TINYVLLM_MODEL"], device="cuda", dtype=torch.bfloat16,
    )
    runners = tuple(
        ModelRunner(model, block_size=16, num_blocks=144,
                    max_model_len=1280, attention_backend="cuda")
        for _ in range(2)
    )
    runners[0].attention_type = PreviousCudaAttention
    for runner in runners:
        runner.kv_cache.key.fill_(float("nan"))
        runner.kv_cache.value.fill_(float("nan"))
    manager = BlockManager(block_size=16, num_blocks=144)
    first = tuple(range(100, 1252))
    second = tuple(range(3000, 4081))
    prefill = WorkClass.PREFILL
    steps = (
        ((10, first, 256, False, prefill), (22, second, 17, False, prefill)),
        ((22, second, 65, False, prefill), (10, first, 256, False, prefill)),
        ((10, first, 512, False, prefill), (22, second, 512, False, prefill)),
        ((22, second, 487, True, prefill), (10, first, 128, True, prefill)),
        ((10, first + (1500,), 1, True, WorkClass.DECODE),
         (22, second + (4500,), 1, True, WorkClass.DECODE)),
    )
    budget = _RealLogitBudget()
    try:
        with torch.inference_mode():
            for step_id, records in enumerate(steps):
                transaction, batch = prepare(
                    manager, step_id, records, max_model_len=1280,
                )
                transaction.commit_before_dispatch()
                expected = runners[0].run(batch)
                actual = runners[1].run(batch)
                if expected is None:
                    assert actual is None
                else:
                    budget(actual, expected)
                transaction.complete_writes()
    finally:
        for runner in runners:
            runner.clear()
        print("extended P3-B1 vs chunk backend:", budget.summary())
    assert budget.rows == 4
