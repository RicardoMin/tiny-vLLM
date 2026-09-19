"""Fresh-prefill contracts and opt-in CUDA attention/cache integration gates."""

from __future__ import annotations

import os

import pytest
import torch

from tinyvllm.batch import PreparedBatch, StepMode
from tinyvllm.paged_kv import PagedKVCache
from tinyvllm.request import SamplingParams
from tinyvllm.runtime.attention import CudaAttention
from tinyvllm.runtime.contracts import ModelInput


def _fresh_batch(lengths=(5, 3), *, block_size=3):
    counts = [(length + block_size - 1) // block_size for length in lengths]
    num_blocks = sum(counts) + 2
    # Reversed nonadjacent first/second halves exercise logical page ordering.
    physical = list(range(num_blocks))[::2][::-1] + list(range(num_blocks))[1::2]
    block_ids = tuple(physical[:sum(counts)])
    offsets, table_offsets, positions, slots, owners = [0], [0], [], [], []

    for index, (length, count) in enumerate(zip(lengths, counts)):
        first = table_offsets[-1]
        table = block_ids[first:first + count]
        slots.extend(
            table[position // block_size] * block_size + position % block_size
            for position in range(length)
        )
        positions.extend(range(length))
        owners.extend([index] * length)
        offsets.append(offsets[-1] + length)
        table_offsets.append(first + count)

    request_ids = tuple(10 + index for index in range(len(lengths)))
    return PreparedBatch(
        plan_id=0,
        mode=StepMode.PREFILL_ONLY,
        num_decode_items=0,
        input_ids=tuple(range(1, sum(lengths) + 1)),
        positions=tuple(positions),
        token_request_slots=tuple(owners),
        q_offsets=tuple(offsets),
        context_lens_before=(0,) * len(lengths),
        query_lens=lengths,
        context_lens_after=lengths,
        sample_token_indices=tuple(end - 1 for end in offsets[1:]),
        sample_request_ids=request_ids,
        sampling_params=(SamplingParams(),) * len(lengths),
        kv_block_size=block_size,
        kv_num_blocks=num_blocks,
        kv_block_table_ids=block_ids,
        kv_block_table_offsets=tuple(table_offsets),
        kv_slot_mapping=tuple(slots),
        request_ids=request_ids,
    )


def test_fresh_lowering_omits_gather_metadata_but_reference_keeps_it(monkeypatch):
    batch = _fresh_batch()
    original_tensor = torch.tensor

    def checked_tensor(data, *args, **kwargs):
        assert data is not batch.kv_block_table_ids, "fresh prefill copied read tables"
        return original_tensor(data, *args, **kwargs)

    def unexpected_arange(*args, **kwargs):
        pytest.fail("fresh prefill generated logical gather positions")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "tensor", checked_tensor)
        patch.setattr(torch, "arange", unexpected_arange)
        optimized = ModelInput.from_prepared_batch(
            batch, device="cpu", paged_decode=True,
        )

    assert optimized.kv.read_slots == (None,) * len(batch.query_lens)
    assert optimized.kv.decode is None
    assert optimized.kv.slot_mapping.tolist() == list(batch.kv_slot_mapping)
    reference = ModelInput.from_prepared_batch(batch, device="cpu")
    assert reference.kv.decode is None
    for index, read_slots in enumerate(reference.kv.read_slots):
        start, end = batch.q_offsets[index:index + 2]
        assert read_slots.tolist() == list(batch.kv_slot_mapping[start:end])


CUDA_ENABLED = (
    os.environ.get("TINYVLLM_RUN_CUDA") == "1"
    and torch.cuda.is_available()
)


@pytest.mark.skipif(not CUDA_ENABLED, reason="requires TINYVLLM_RUN_CUDA=1 and CUDA")
@pytest.mark.parametrize(
    "dtype,lengths,heads,kv_heads,head_dim,block_size",
    [
        (torch.bfloat16, (17, 31), 16, 8, 128, 16),
        (torch.float16, (5, 3), 4, 2, 32, 3),
        (torch.bfloat16, (7, 2), 2, 2, 6, 4),
    ],
    ids=["qwen3-bf16-pages", "fp16-gqa-small-pages", "bf16-mha-small-head"],
)
@torch.inference_mode()
def test_fresh_prefill_strided_values_are_causal_and_write_exact_cache(
    dtype, lengths, heads, kv_heads, head_dim, block_size, monkeypatch,
):
    batch = _fresh_batch(lengths, block_size=block_size)
    data = ModelInput.from_prepared_batch(batch, device="cuda", paged_decode=True)
    generator = torch.Generator().manual_seed(381)
    widths = (heads * head_dim, kv_heads * head_dim, kv_heads * head_dim)
    packed = torch.randn(
        sum(lengths), sum(widths), generator=generator,
    ).to(device="cuda", dtype=dtype)
    query, key, value = packed.split(widths, dim=-1)
    query = query.reshape(-1, heads, head_dim).contiguous()
    key = key.reshape(-1, kv_heads, head_dim).contiguous()
    value = value.reshape(-1, kv_heads, head_dim)
    assert value.stride(0) == sum(widths)
    assert not value.is_contiguous()

    cache = PagedKVCache(
        num_layers=2,
        num_blocks=batch.kv_num_blocks,
        block_size=block_size,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        device="cuda",
        dtype=dtype,
    )
    cache.key.fill_(float("nan"))
    cache.value.fill_(float("nan"))
    expected_key = cache.key.cpu()
    expected_value = cache.value.cpu()
    cpu_slots = torch.tensor(batch.kv_slot_mapping)
    expected_key[1].index_copy_(0, cpu_slots, key.cpu())
    expected_value[1].index_copy_(0, cpu_slots, value.cpu())

    def forbidden_cache_operation(*args, **kwargs):
        pytest.fail("fresh CUDA prefill used reference KV write or gather")

    monkeypatch.setattr(cache, "read", forbidden_cache_operation)
    monkeypatch.setattr(cache, "write", forbidden_cache_operation)
    attention = CudaAttention(batch, data, cache)
    scale = head_dim**-0.5
    actual = attention(1, query, key, value, scale=scale)

    # Explicit FP32 QK/softmax/PV oracle; no production SDPA or page lookup.
    expected = []
    for start, end in zip(batch.q_offsets, batch.q_offsets[1:]):
        q = query[start:end].float().cpu().transpose(0, 1)
        k = key[start:end].float().cpu().repeat_interleave(heads // kv_heads, dim=1)
        v = value[start:end].float().cpu().repeat_interleave(heads // kv_heads, dim=1)
        scores = (q @ k.transpose(0, 1).transpose(-1, -2)) * scale
        future = torch.ones(end - start, end - start, dtype=torch.bool).triu(1)
        scores.masked_fill_(future, -float("inf"))
        expected.append((scores.softmax(-1) @ v.transpose(0, 1)).transpose(0, 1))

    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(
        actual.float().cpu(), torch.cat(expected), atol=tolerance, rtol=tolerance,
    )
    torch.testing.assert_close(
        cache.key.cpu(), expected_key, atol=0, rtol=0, equal_nan=True,
    )
    torch.testing.assert_close(
        cache.value.cpu(), expected_value, atol=0, rtol=0, equal_nan=True,
    )

    # Perturb a future V row: earlier queries and the other request cannot see it.
    changed_row = lengths[0] - 1
    value[changed_row].add_(8)
    changed = attention(1, query, key, value, scale=scale)
    untouched_rows = torch.arange(sum(lengths), device="cuda") != changed_row
    torch.testing.assert_close(
        changed[untouched_rows], actual[untouched_rows], atol=0, rtol=0,
    )
    assert not torch.equal(changed[changed_row], actual[changed_row])
