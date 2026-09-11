from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from tinyvllm.batch import PreparedBatch, StepPlan
from tinyvllm.contiguous_kv import ContiguousKVCache
from tinyvllm.runtime.runner import ModelRunner

from test_batched_executor import (
    MAX_MODEL_LEN,
    decode_item,
    make_request,
    prefill_item,
    prepare,
    tiny_model,
)


def mixed_batches() -> tuple[PreparedBatch, ...]:
    first = make_request(11, (1, 2, 3), output=(7, 11))
    second = make_request(22, (4, 5, 6, 8, 9))
    third = make_request(33, (12, 13), output=(17,))

    warmup = prepare(
        StepPlan(
            plan_id=1,
            items=(
                prefill_item(first, slot=4),
                prefill_item(
                    second,
                    slot=8,
                    query_len=2,
                    should_sample=False,
                ),
            ),
        ),
        (first, second),
    )
    mixed = prepare(
        StepPlan(
            plan_id=2,
            items=(
                decode_item(
                    first,
                    slot=4,
                    token_start=3,
                    context_len_before=3,
                ),
                prefill_item(
                    second,
                    slot=8,
                    token_start=2,
                    query_len=2,
                    context_len_before=2,
                    should_sample=False,
                ),
                prefill_item(third, slot=12),
            ),
        ),
        (first, second, third),
    )
    reordered = prepare(
        StepPlan(
            plan_id=3,
            items=(
                decode_item(
                    third,
                    slot=12,
                    token_start=2,
                    context_len_before=2,
                ),
                decode_item(
                    first,
                    slot=4,
                    token_start=4,
                    context_len_before=4,
                ),
                prefill_item(
                    second,
                    slot=8,
                    token_start=4,
                    query_len=1,
                    context_len_before=4,
                ),
            ),
        ),
        (third, first, second),
    )
    return warmup, mixed, reordered


def make_cache(model) -> ContiguousKVCache:
    config = model.config
    parameter = next(model.parameters())
    return ContiguousKVCache(
        num_layers=config.num_hidden_layers,
        max_seq_len=MAX_MODEL_LEN,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def test_mixed_runner_matches_single_request_hidden_logits_and_kv() -> None:
    model = tiny_model().eval()
    oracle = deepcopy(model)
    runner = ModelRunner(model, max_model_len=MAX_MODEL_LEN)
    oracle_caches = {}
    captured = []
    handle = model.model.register_forward_hook(
        lambda module, args, output: captured.append(
            output.detach().clone()
        )
    )

    try:
        with torch.inference_mode():
            for batch in mixed_batches():
                captured.clear()
                actual_logits = runner.run(batch)
                expected_parts = []

                for index, request_id in enumerate(batch.request_ids):
                    if request_id not in oracle_caches:
                        oracle_caches[request_id] = make_cache(oracle)
                    start, end = batch.q_offsets[index:index + 2]
                    expected_parts.append(
                        oracle(
                            torch.tensor(batch.input_ids[start:end]),
                            torch.tensor(batch.positions[start:end]),
                            cache=oracle_caches[request_id],
                        )
                    )

                    actual_cache = runner._caches[request_id]
                    expected_cache = oracle_caches[request_id]
                    assert actual_cache.length == batch.context_lens_after[index]
                    assert actual_cache.length == expected_cache.length
                    for actual, expected in (
                        (actual_cache.key, expected_cache.key),
                        (actual_cache.value, expected_cache.value),
                    ):
                        torch.testing.assert_close(
                            actual[:, :actual_cache.length],
                            expected[:, :expected_cache.length],
                            atol=1e-5,
                            rtol=1e-5,
                        )

                expected_hidden = torch.cat(expected_parts)
                assert len(captured) == 1
                torch.testing.assert_close(
                    captured[0],
                    expected_hidden,
                    atol=1e-5,
                    rtol=1e-5,
                )
                expected_logits = oracle.compute_logits(
                    expected_hidden,
                    torch.tensor(
                        batch.sample_token_indices,
                        dtype=torch.long,
                    ),
                )
                assert actual_logits is not None
                torch.testing.assert_close(
                    actual_logits,
                    expected_logits,
                    atol=1e-5,
                    rtol=1e-5,
                )
    finally:
        handle.remove()


def test_runner_batches_every_projection_and_selects_only_sample_rows(
    monkeypatch,
) -> None:
    model = tiny_model()
    runner = ModelRunner(model, max_model_len=MAX_MODEL_LEN)
    warmup, batch, _ = mixed_batches()
    runner.run(warmup)
    calls = {}
    handles = []

    def record_rows(name):
        def hook(module, args):
            calls[name].append(args[0].shape[0])
        return hook

    modules = {"model": model}
    for index, layer in enumerate(model.model.layers):
        modules.update(
            {
                f"qkv_{index}": layer.self_attn.qkv_proj,
                f"o_{index}": layer.self_attn.o_proj,
                f"gate_up_{index}": layer.mlp.gate_up_proj,
                f"down_{index}": layer.mlp.down_proj,
            }
        )
    for name, module in modules.items():
        calls[name] = []
        handles.append(
            module.register_forward_pre_hook(record_rows(name))
        )

    logits_calls = []
    original = model.compute_logits

    def record_logits(hidden_states, token_indices=None):
        assert token_indices is not None
        logits_calls.append(
            (hidden_states.shape[0], token_indices.tolist())
        )
        return original(hidden_states, token_indices)

    monkeypatch.setattr(model, "compute_logits", record_logits)
    try:
        logits = runner.run(batch)
    finally:
        for handle in handles:
            handle.remove()

    assert all(
        rows == [batch.num_query_tokens]
        for rows in calls.values()
    )
    assert logits_calls == [
        (batch.num_query_tokens, list(batch.sample_token_indices)),
    ]
    assert batch.sample_token_indices == (0, 4)
    assert logits is not None
    assert logits.shape == (2, model.config.vocab_size)


def test_runner_skips_logits_when_no_item_samples(monkeypatch) -> None:
    model = tiny_model()
    runner = ModelRunner(model, max_model_len=MAX_MODEL_LEN)
    batch = replace(
        mixed_batches()[0],
        sample_token_indices=(),
        sample_request_ids=(),
        sampling_params=(),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("partial prefill must not compute logits")

    monkeypatch.setattr(model, "compute_logits", forbidden)
    assert runner.run(batch) is None
    assert {
        key: cache.length
        for key, cache in runner._caches.items()
    } == {
        11: 3,
        22: 2,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"q_offsets": (0, 1, 2, 5)},
        {"context_lens_after": (4, 5, 2)},
        {"positions": (3, 2, 8, 0, 1)},
        {"sample_token_indices": (0, 3)},
        {"sample_request_ids": (33, 11)},
    ],
)
def test_runner_rejects_invalid_metadata_before_any_kv_write(
    updates,
    monkeypatch,
) -> None:
    runner = ModelRunner(tiny_model(), max_model_len=MAX_MODEL_LEN)
    warmup, batch, _ = mixed_batches()
    runner.run(warmup)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid batch must be rejected before KV writes")

    monkeypatch.setattr(ContiguousKVCache, "write", forbidden)
    with pytest.raises(ValueError):
        runner.run(replace(batch, **updates))
    assert runner._caches[11].length == 3
    assert runner._caches[22].length == 2


def test_runner_checks_all_cache_lengths_before_any_kv_write(
    monkeypatch,
) -> None:
    runner = ModelRunner(tiny_model(), max_model_len=MAX_MODEL_LEN)
    warmup, batch, _ = mixed_batches()
    runner.run(warmup)
    runner._caches[22].clear()

    def forbidden(*args, **kwargs):
        pytest.fail("second cache mismatch must precede first cache write")

    monkeypatch.setattr(ContiguousKVCache, "write", forbidden)
    with pytest.raises(RuntimeError, match="cache length"):
        runner.run(batch)
    assert runner._caches[11].length == 3
    assert runner._caches[22].length == 0


@pytest.mark.parametrize("failure_point", ("layer", "norm"))
def test_model_failure_does_not_advance_any_cache(
    failure_point,
    monkeypatch,
) -> None:
    model = tiny_model()
    runner = ModelRunner(model, max_model_len=MAX_MODEL_LEN)
    warmup, batch, _ = mixed_batches()
    runner.run(warmup)
    prefixes = {
        request_id: (
            cache.key[:, :cache.length].clone(),
            cache.value[:, :cache.length].clone(),
        )
        for request_id, cache in runner._caches.items()
    }

    def fail(*args, **kwargs):
        raise RuntimeError("injected model failure")

    target = (
        model.model.layers[-1]
        if failure_point == "layer"
        else model.model.norm
    )
    monkeypatch.setattr(target, "forward", fail)
    with pytest.raises(RuntimeError, match="injected model failure"):
        runner.run(batch)

    assert runner._caches[11].length == 3
    assert runner._caches[22].length == 2
    assert runner._caches[33].length == 0
    for request_id, (key, value) in prefixes.items():
        cache = runner._caches[request_id]
        torch.testing.assert_close(cache.key[:, :cache.length], key)
        torch.testing.assert_close(cache.value[:, :cache.length], value)