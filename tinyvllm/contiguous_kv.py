from __future__ import annotations

import torch


class ContiguousKVCache:
    """Layer-major contiguous KV cache for one request."""

    def __init__(
        self,
        *,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if min(
            num_layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
        ) <= 0:
            raise ValueError("cache dimensions must be positive")

        shape = (
            num_layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
        )

        self.key = torch.empty(
            shape,
            device=device,
            dtype=dtype,
        )
        self.value = torch.empty_like(self.key)

        self.max_seq_len = max_seq_len
        self._length = 0


    @property
    def length(self) -> int:
        return self._length


    def write(
        self,
        layer_idx: int,
        start: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if start != self._length:
            raise ValueError(
                "KV write must begin at current cache length"
            )

        if (
            key.ndim != 3
            or key.shape != value.shape
            or tuple(key.shape[1:]) != tuple(self.key.shape[2:])
        ):
            raise ValueError("invalid KV tensor shape")

        end = start + key.shape[0]

        if end > self.max_seq_len:
            raise ValueError("KV write exceeds cache capacity")

        self.key[layer_idx, start:end].copy_(key)
        self.value[layer_idx, start:end].copy_(value)


    def read(
        self,
        layer_idx: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= end <= self.max_seq_len:
            raise ValueError("invalid KV read range")

        return (
            self.key[layer_idx, :end],
            self.value[layer_idx, :end],
        )


    def advance(self, length: int) -> None:
        if not self._length <= length <= self.max_seq_len:
            raise ValueError("invalid KV cache length")

        self._length = length

    def clear(self) -> None:
        self._length = 0