from __future__ import annotations

import torch

from tinyvllm.kernels._extension import load_cuda_kernels
from tinyvllm.paged_kv import PagedKVCache


def store_kv(
    cache: PagedKVCache,
    layer_idx: int,
    slots: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Scatter both tensors in one launch; CPU validation owns slot values."""
    cache._check_indices(layer_idx, slots)
    load_cuda_kernels()

    torch.ops.tinyvllm_cuda.store_kv(
        key,
        value,
        cache.key[layer_idx],
        cache.value[layer_idx],
        slots,
    )