from __future__ import annotations

from collections.abc import Iterable

import torch

from tinyvllm._validation import (
    _check_non_negative_int,
    _check_positive_int,
)
from tinyvllm.cache import BlockRef

class PagedKVCache:
    """Physical K/V storage addressed by logical page tables.

    The pool owns page lifetime. This class owns only device tensors and
    translates logical token positions into flat physical slots.
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


    def logical_slots(
        self,
        block_table: Iterable[BlockRef],
        *,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Lower logical positions [start, end) to physical slot IDs."""
        _check_non_negative_int("start", start)
        _check_non_negative_int("end", end)

        if end < start:
            raise ValueError("end must not be less than start")

        if end == start:
            return torch.empty(
                0,
                device=self.device,
                dtype=torch.long,
            )

        refs = tuple(block_table)

        required_blocks = (
            end + self.block_size -1
        ) // self.block_size

        if required_blocks > len(refs):
            raise ValueError(
                "page table is too short for the requested range"
            )

        block_ids: list[int] = []

        for ref in refs[:required_blocks]:
            if not isinstance(ref, BlockRef):
                raise TypeError(
                    "block_table must contain BlockRef values"
                )

            if not 0 <= ref.block_id < self.num_blocks:
                raise ValueError(
                    "page reference is outside this cache"
                )

            block_ids.append(ref.block_id)

        positions = torch.arange(
            start,
            end,
            device=self.device,
            dtype=torch.long,
        )

        physical_blocks = torch.tensor(
            block_ids,
            device=self.device,
            dtype=torch.long,
        ).index_select(
            0,
            positions // self.block_size,
        )

        return physical_blocks * self.block_size + positions.remainder(
            self.block_size
        )


    def write(
        self,
        layer_idx: int,
        slots: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write one layer's K/V rows into physical slots."""
        self._check_layer(layer_idx)
        self._check_slots(slots)

        expected_shape = (
            slots.numel(),
            self.num_kv_heads,
            self.head_dim,
        )

        if key.shape != expected_shape or value.shape != expected_shape:
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
        """Read rows in logical order represented by the slot mapping."""
        self._check_layer(layer_idx)
        self._check_slots(slots)

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


    def _check_layer(self, layer_idx: int) -> None:
        _check_non_negative_int("layer_idx", layer_idx)

        if layer_idx >= self.num_layers:
            raise ValueError("layer_idx is out of range")


    def _check_slots(self, slots: torch.Tensor) -> None:
        if not isinstance(slots, torch.Tensor):
            raise TypeError("slots must be a torch.Tensor")

        if slots.ndim != 1 or slots.dtype is not torch.long:
            raise ValueError(
                "slots must be a one-dimensional int64 tensor"
            )

        if slots.device != self.device:
            raise ValueError(
                "slots must use the cache device"
            )

        if slots.numel():
            if bool(torch.any(slots < 0)) or bool(
                torch.any(slots >= self.num_slots)
            ):
                raise ValueError(
                    "slots are out of range"
                )
