"""Shared scattered-page inputs and independent FP32 dense attention oracle."""

import torch

from tinyvllm.paged_kv import PagedKVCache
from tinyvllm.runtime.attention import CudaAttention
from tinyvllm.runtime.contracts import ModelInput


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
