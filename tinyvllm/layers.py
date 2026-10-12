from __future__ import annotations

from collections.abc import Sequence

import math

import torch
import torch.nn.functional as F
from torch import nn

from tinyvllm._validation import _check_positive_int
from tinyvllm.kernels.pointwise import rms_norm, silu_and_mul, apply_rope


class PackedLinear(nn.Module):
    """A bias-free linear layer with several row-packed logical shards."""

    def __init__(
        self,
        input_size: int,
        output_sizes: Sequence[int],
        *,
        shard_names: Sequence[str] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        _check_positive_int("input_size", input_size)
        self.input_size = input_size
        self.output_sizes = tuple(output_sizes)

        if not self.output_sizes:
            raise ValueError("output_sizes must be non-empty")

        for size in self.output_sizes:
            _check_positive_int("output_size", size)

        if shard_names is None:
            shard_names = tuple(
                str(index)
                for index in range(len(self.output_sizes))
            )
        else:
            shard_names = tuple(shard_names)

        if len(shard_names) != len(self.output_sizes):
            raise ValueError("shard_names must match output_sizes")

        if len(set(shard_names)) != len(shard_names):
            raise ValueError("shard_names must be unique")

        self.shard_names = shard_names
        self._name_to_index = {
            name: index
            for index, name in enumerate(shard_names)
        }

        offsets: list[int] = []
        offset = 0

        for size in self.output_sizes:
            offsets.append(offset)
            offset += size

        self._offsets = tuple(offsets)

        self.weight = nn.Parameter(
            torch.empty(
                offset,
                self.input_size,
                device=device,
                dtype=dtype,
            )
        )

        self.reset_parameters()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
        )

    def load_shard(
        self,
        shard: str | int,
        loaded_weight: torch.Tensor,
    ) -> None:
        if isinstance(shard, str):
            try:
                index = self._name_to_index[shard]
            except KeyError as exc:
                raise KeyError(f"unknown shard: {shard}") from exc
        else:
            index = shard

        if not isinstance(index, int) or not 0 <= index < len(self.output_sizes):
            raise IndexError(f"shard index out of range: {index}")

        start = self._offsets[index]
        size = self.output_sizes[index]

        expected_shape = (
            size,
            self.input_size,
        )

        if tuple(loaded_weight.shape) != expected_shape:
            raise ValueError(
                f"expected shard shape {expected_shape}, "
                f"got {tuple(loaded_weight.shape)}"
            )

        with torch.no_grad():
            self.weight[
                start : start + size
            ].copy_(loaded_weight)


class RMSNorm(nn.Module):
    """Qwen3 RMSNorm with float32 variance accumulation."""

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(
                hidden_size,
                device=device,
                dtype=dtype,
            )
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    """Half-split rotary position embedding for flat token positions."""

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        *,
        base: float = 1_000_000.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

        if max_position <= 0:
            raise ValueError("max_position must be positive")

        if base <= 0:
            raise ValueError("base must be positive")

        if head_dim % 2:
            raise ValueError("head_dim must be even")

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                    device=device,
                )
                / head_dim
            )
        )

        positions = torch.arange(
            max_position,
            dtype=torch.float32,
            device=device,
        )

        frequencies = torch.outer(
            positions,
            inv_freq,
        )

        self.register_buffer(
            "cos",
            frequencies.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sin",
            frequencies.sin(),
            persistent=False,
        )


    def forward(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather and cast one pair for all attention layers in this forward."""
        cos = self.cos.index_select(
            0,
            positions,
        ).to(dtype).unsqueeze(1)

        sin = self.sin.index_select(
            0,
            positions,
        ).to(dtype).unsqueeze(1)

        return cos, sin


    @staticmethod
    def rotate(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return apply_rope(x, cos, sin)


class SiluAndMul(nn.Module):
    """Compute silu(gate) * up for a packed MLP activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)