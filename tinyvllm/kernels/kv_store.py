from __future__ import annotations

import torch

from tinyvllm.kernels.paged_decode import load_paged_decode
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
    load_paged_decode()

    torch.ops.tinyvllm_cuda.store_kv(
        key,
        value,
        cache.key[layer_idx],
        cache.value[layer_idx],
        slots,
    )