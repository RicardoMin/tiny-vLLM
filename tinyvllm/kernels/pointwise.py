from __future__ import annotations

import torch

from tinyvllm.kernels._extension import load_cuda_kernels

def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize strided 2D/3D CUDA rows without packing a temporary input."""
    load_cuda_kernels()
    return torch.ops.tinyvllm_cuda.rms_norm(x, weight, eps)

    
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fuse the activation and gate/up product, preserving intermediate rounding."""
    load_cuda_kernels()
    return torch.ops.tinyvllm_cuda.silu_and_mul(x)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply half-split RoPE with the original intermediate rounding."""
    load_cuda_kernels()
    return torch.ops.tinyvllm_cuda.apply_rope(x, cos, sin)