"""Lower-right chunk causality and allocation-safe fallback regression gates."""

from __future__ import annotations

import os

import pytest
import torch
from torch.backends.cuda import SDPAParams, can_use_flash_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import CausalBias

from tinyvllm.batch import PreparedBatch, StepMode
from tinyvllm.paged_kv import PagedKVCache
from tinyvllm.request import SamplingParams
from tinyvllm.runtime import attention as attention_module
from tinyvllm.runtime.attention import CudaAttention, ReferenceAttention
from tinyvllm.runtime.contracts import ModelInput


def _chunk_batch(before, query_len, *, block_size=16):
    # One continuing chunk and one fresh request, in scattered physical pages.
    befores, lengths = (before, 0), (query_len, 3)
    afters = tuple(a + b for a, b in zip(befores, lengths))
    counts = tuple((length + block_size - 1) // block_size for length in afters)
    num_blocks = sum(counts) + 2
    physical = list(range(num_blocks))[::2][::-1] + list(range(num_blocks))[1::2]
    tables, positions, owners, slots = [], [], [], []
    offsets, table_offsets = [0], [0]

    for index, (start, end, count) in enumerate(zip(befores, afters, counts)):
        table = physical[table_offsets[-1]:table_offsets[-1] + count]
        tables.extend(table)
        table_offsets.append(len(tables))
        positions.extend(range(start, end))
        owners.extend([index] * (end - start))
        slots.extend(
            table[position // block_size] * block_size + position % block_size
            for position in range(start, end)
        )
        offsets.append(offsets[-1] + end - start)

    return PreparedBatch(
        plan_id=0,
        mode=StepMode.PREFILL_ONLY,
        num_decode_items=0,
        input_ids=tuple(range(1, sum(lengths) + 1)),
        positions=tuple(positions),
        token_request_slots=tuple(owners),
        q_offsets=tuple(offsets),
        context_lens_before=befores,
        query_lens=lengths,
        context_lens_after=afters,
        sample_token_indices=tuple(end - 1 for end in offsets[1:]),
        sample_request_ids=(10, 20),
        sampling_params=(SamplingParams(), SamplingParams()),
        kv_block_size=block_size,
        kv_num_blocks=num_blocks,
        kv_block_table_ids=tuple(tables),
        kv_block_table_offsets=tuple(table_offsets),
        kv_slot_mapping=tuple(slots),
        request_ids=(10, 20),
    )


def _inputs(batch, *, device, dtype, heads, kv_heads, head_dim):
    generator = torch.Generator().manual_seed(2081)
    query = torch.randn(
        batch.num_query_tokens, heads, head_dim, generator=generator,
    ).to(device=device, dtype=dtype)
    logical_keys, logical_values = [], []
    new_keys, new_values = [], []
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=batch.kv_num_blocks,
        block_size=batch.kv_block_size,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
    cache.key.fill_(float("nan"))
    cache.value.fill_(float("nan"))

    for index, (before, after) in enumerate(zip(
        batch.context_lens_before, batch.context_lens_after,
    )):
        key = torch.randn(after, kv_heads, head_dim, generator=generator).to(dtype)
        value = torch.randn(after, kv_heads, head_dim, generator=generator).to(dtype)
        logical_keys.append(key)
        logical_values.append(value)
        new_keys.append(key[before:])
        new_values.append(value[before:])
        first, last = batch.kv_block_table_offsets[index:index + 2]
        table = batch.kv_block_table_ids[first:last]
        old_slots = torch.tensor([
            table[position // batch.kv_block_size] * batch.kv_block_size
            + position % batch.kv_block_size
            for position in range(before)
        ], device=device, dtype=torch.long)
        for layer in range(2):
            cache.key[layer].index_copy_(0, old_slots, key[:before].to(device))
            cache.value[layer].index_copy_(0, old_slots, value[:before].to(device))

    key = torch.cat(new_keys).to(device)
    # Match the noncontiguous token stride of a value slice from fused QKV.
    packed = torch.empty(
        batch.num_query_tokens, 2 * kv_heads, head_dim, device=device, dtype=dtype,
    )
    value = packed[:, kv_heads:]
    value.copy_(torch.cat(new_values))
    data = ModelInput.from_prepared_batch(batch, device=device, paged_decode=True)
    return CudaAttention(batch, data, cache), query, key, value, logical_keys, logical_values


def _oracle(batch, query, logical_keys, logical_values, scale):
    outputs = []
    for index, (key, value) in enumerate(zip(logical_keys, logical_values)):
        start, end = batch.q_offsets[index:index + 2]
        q = query[start:end].float().cpu().transpose(0, 1)
        repeats = q.shape[0] // key.shape[1]
        k = key.float().repeat_interleave(repeats, dim=1).transpose(0, 1)
        v = value.float().repeat_interleave(repeats, dim=1).transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) * scale
        positions = torch.arange(batch.context_lens_before[index], key.shape[0])
        future = torch.arange(key.shape[0])[None, :] > positions[:, None]
        scores.masked_fill_(future, -float("inf"))
        outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
    return torch.cat(outputs)


@torch.inference_mode()
def test_chunk_cpu_fallback_reuses_mask_and_preserves_reference(monkeypatch):
    batch = _chunk_batch(5, 3, block_size=3)
    attention, q, k, v, keys, values = _inputs(
        batch, device="cpu", dtype=torch.float32, heads=4, kv_heads=2, head_dim=8,
    )
    assert isinstance(attention.masks[0], CausalBias)
    assert attention.masks[1] is None
    reference = ReferenceAttention(
        batch, ModelInput.from_prepared_batch(batch, device="cpu"), attention.cache,
    )
    assert type(reference.masks[0]) is torch.Tensor

    # CPU coverage exercises actual gather + SDPA; only the CUDA store is replaced.
    monkeypatch.setattr(
        attention, "_write_kv",
        lambda layer, key, value: attention.cache.write(
            layer, attention.kv.slot_mapping, key, value,
        ),
    )
    monkeypatch.setattr(attention_module, "can_use_flash_attention", lambda params: False)

    def forbidden_materialize(*args, **kwargs):
        pytest.fail("CausalBias fell back by materializing inside each SDPA call")

    monkeypatch.setattr(CausalBias, "_materialize", forbidden_materialize)
    expected = _oracle(batch, q, keys, values, 8**-0.5)
    actual = attention(0, q, k, v, scale=8**-0.5)
    cached_mask = attention.masks[0]
    assert type(cached_mask) is torch.Tensor
    assert cached_mask.dtype == torch.bool
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    monkeypatch.setattr(attention_module, "can_use_flash_attention", forbidden_materialize)
    repeated = attention(1, q, k, v, scale=8**-0.5)
    assert attention.masks[0] is cached_mask
    torch.testing.assert_close(repeated, expected, atol=1e-6, rtol=1e-5)


CUDA_ENABLED = (
    os.environ.get("TINYVLLM_RUN_CUDA") == "1" and torch.cuda.is_available()
)
cuda_gate = pytest.mark.skipif(
    not CUDA_ENABLED, reason="requires TINYVLLM_RUN_CUDA=1 and CUDA",
)


@cuda_gate
@pytest.mark.parametrize(
    "before,query_len,dtype,heads,kv_heads,head_dim",
    [
        (15, 2, torch.bfloat16, 16, 8, 128),
        (16, 17, torch.float16, 4, 2, 32),
        (17, 65, torch.bfloat16, 2, 2, 6),
    ],
    ids=["bf16-gqa-page-tail", "fp16-gqa-page-boundary", "bf16-small-head-long-chunk"],
)
@torch.inference_mode()
def test_chunk_flash_matches_independent_oracle_and_causality(
    before, query_len, dtype, heads, kv_heads, head_dim, monkeypatch,
):
    batch = _chunk_batch(before, query_len)
    attention, q, k, v, keys, values = _inputs(
        batch, device="cuda", dtype=dtype, heads=heads, kv_heads=kv_heads,
        head_dim=head_dim,
    )
    params = SDPAParams(
        q[:query_len].transpose(0, 1)[None],
        keys[0].to("cuda").transpose(0, 1)[None],
        values[0].to("cuda").transpose(0, 1)[None],
        None, 0.0, False, heads != kv_heads,
    )
    supported = can_use_flash_attention(params)

    def forbidden_materialize(*args, **kwargs):
        pytest.fail("chunk attention materialized a symbolic mask in SDPA")

    monkeypatch.setattr(CausalBias, "_materialize", forbidden_materialize)
    scale = head_dim**-0.5
    actual = attention(0, q, k, v, scale=scale)
    assert isinstance(attention.masks[0], CausalBias) == supported
    expected = _oracle(batch, q, keys, values, scale)
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(
        actual.float().cpu(), expected, atol=tolerance, rtol=tolerance,
    )

    # The last chunk token is future for all earlier queries and a separate request.
    v[query_len - 1].add_(8)
    changed = attention(0, q, k, v, scale=scale)
    untouched = torch.arange(batch.num_query_tokens, device="cuda") != query_len - 1
    torch.testing.assert_close(changed[untouched], actual[untouched], atol=0, rtol=0)
    assert not torch.equal(changed[query_len - 1], actual[query_len - 1])


@cuda_gate
@torch.inference_mode()
def test_chunk_math_fallback_uses_lower_right_and_reuses_mask(monkeypatch):
    batch = _chunk_batch(5, 3, block_size=3)
    attention, q, k, v, _, _ = _inputs(
        batch, device="cuda", dtype=torch.bfloat16, heads=4, kv_heads=2, head_dim=8,
    )
    q.zero_()
    k.zero_()
    # Uniform attention over logical values 0..7 distinguishes left/right alignment.
    table = batch.kv_block_table_ids[:batch.kv_block_table_offsets[1]]
    for position in range(5):
        slot = table[position // 3] * 3 + position % 3
        attention.cache.key[:, slot].zero_()
        attention.cache.value[:, slot].fill_(position)
    v[:3].copy_(torch.arange(5, 8, device="cuda", dtype=v.dtype)[:, None, None])

    def forbidden_materialize(*args, **kwargs):
        pytest.fail("MATH fallback rematerialized CausalBias inside SDPA")

    monkeypatch.setattr(CausalBias, "_materialize", forbidden_materialize)
    first = attention(0, q, k, v, scale=8**-0.5)

    # Backend settings can change between layers: recheck a still-symbolic mask.
    with sdpa_kernel(SDPBackend.MATH):
        second = attention(1, q, k, v, scale=8**-0.5)
        cached_mask = attention.masks[0]
        repeated = attention(0, q, k, v, scale=8**-0.5)

    assert type(cached_mask) is torch.Tensor
    assert attention.masks[0] is cached_mask
    expected = torch.tensor([2.5, 3.0, 3.5], device="cuda", dtype=q.dtype)
    torch.testing.assert_close(first[:3], expected[:, None, None].expand_as(first[:3]))
    torch.testing.assert_close(second[:3], expected[:, None, None].expand_as(second[:3]))
    torch.testing.assert_close(second, first, atol=0.02, rtol=0.02)
    torch.testing.assert_close(repeated, second, atol=0, rtol=0)
