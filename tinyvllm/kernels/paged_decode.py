from __future__ import annotations

import torch

from tinyvllm.paged_kv import PagedKVCache
from tinyvllm.runtime.contracts import SingleQueryMetadata
from tinyvllm.kernels._extension import load_cuda_kernels


def paged_decode_attention(
    q: torch.Tensor,
    cache: PagedKVCache,
    layer_idx: int,
    metadata: SingleQueryMetadata,
    output: torch.Tensor,
    *,
    scale: float,
) -> None:
    """Fill selected query rows on PyTorch's current CUDA stream."""
    load_cuda_kernels()

    torch.ops.tinyvllm_cuda.paged_decode(
        q,
        cache.key[layer_idx],
        cache.value[layer_idx],
        metadata.query_rows,
        metadata.block_table_ids,
        metadata.block_table_offsets,
        metadata.context_lens,
        output,
        scale,
        cache.block_size,
    )