from __future__ import annotations

import torch

from tinyvllm._validation import _check_non_negative_int
from tinyvllm.kernels._extension import load_cuda_kernels
from tinyvllm.paged_kv import PagedKVCache


def store_kv(
    cache: PagedKVCache,
    layer_idx: int,
    slots: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Scatter CPU-validated real slots; -1 skips internal graph padding."""
    _check_non_negative_int("layer_idx", layer_idx)
    if layer_idx >= cache.num_layers:
        raise ValueError("layer_idx is out of range")

    # Tensor metadata is validated once at the native CUDA boundary.
    load_cuda_kernels()

    torch.ops.tinyvllm_cuda.store_kv(
        key,
        value,
        cache.key[layer_idx],
        cache.value[layer_idx],
        slots,
    )