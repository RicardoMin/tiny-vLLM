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

    
def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized and rounded residual sums without modifying inputs."""
    load_cuda_kernels()
    return torch.ops.tinyvllm_cuda.add_rms_norm(x, residual, weight, eps)

    
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


def qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_eps: float = 1e-6,
    key_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and rotate strided Q/K rows, preserving each rounding step."""
    load_cuda_kernels()

    return torch.ops.tinyvllm_cuda.qk_norm_rope(
        query,
        key,
        query_weight,
        key_weight,
        cos,
        sin,
        query_eps,
        key_eps,
    )