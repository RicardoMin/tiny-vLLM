from __future__ import annotations

import torch

from tinyvllm._validation import (
    _check_non_negative_int,
    _check_positive_int,
)

class PagedKVCache:
    """Device storage only. Slots must come from a validated CPU batch.

    Range, uniqueness and ownership are checked before lowering. This internal
    tensor interface checks shapes/devices without synchronizing GPU scalars.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        for name, value in (
            ("num_layers", num_layers),
            ("num_blocks", num_blocks),
            ("block_size", block_size),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
        ):
            _check_positive_int(name, value)

        self.device = torch.device(device)

        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )

        self.dtype = dtype
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_slots = num_blocks * block_size

        shape = (
            num_layers,
            self.num_slots,
            num_kv_heads,
            head_dim,
        )

        self.key = torch.empty(
            shape,
            device=self.device,
            dtype=dtype,
        )
        self.value = torch.empty_like(self.key)


    def write(
        self,
        layer_idx: int,
        slots: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._check_indices(layer_idx, slots)

        expected = (
            slots.numel(),
            self.num_kv_heads,
            self.head_dim,
        )

        if key.shape != expected or value.shape != expected:
            raise ValueError(
                "K/V rows do not match the slot mapping"
            )

        if key.device != self.device or value.device != self.device:
            raise ValueError(
                "K/V tensors must use the cache device"
            )

        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError(
                "K/V tensors must use the cache dtype"
            )

        with torch.no_grad():
            self.key[layer_idx].index_copy_(
                0,
                slots,
                key,
            )
            self.value[layer_idx].index_copy_(
                0,
                slots,
                value,
            )


    def read(
        self,
        layer_idx: int,
        slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_indices(layer_idx, slots)

        return (
            self.key[layer_idx].index_select(
                0,
                slots,
            ),
            self.value[layer_idx].index_select(
                0,
                slots,
            )
        )


    def _check_indices(self, layer_idx: int, slots: torch.Tensor) -> None:
        _check_non_negative_int("layer_idx", layer_idx)
        if layer_idx >= self.num_layers:
            raise ValueError("layer_idx is out of range")
        if not isinstance(slots, torch.Tensor):
            raise TypeError("slots must be a torch.Tensor")
        if slots.ndim != 1 or slots.dtype is not torch.long:
            raise ValueError("slots must be a one-dimensional int64 tensor")
        if slots.device != self.device:
            raise ValueError("slots must use the cache device")
