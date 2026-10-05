from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import subprocess

import torch


@lru_cache(maxsize=1)
def load_cuda_kernels() -> None:
    """Build once before execution; PyTorch caches the resulting shared library."""
    from torch.utils.cpp_extension import CUDA_HOME, load


    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA backend requires a CUDA toolkit; set CUDA_HOME"
        )

    version = subprocess.check_output(
        [str(Path(CUDA_HOME) / "bin" / "nvcc"), "--version"],
        text=True,
    )

    if f"release {torch.version.cuda}," not in version:
        raise RuntimeError(
            f"Select a CUDA {torch.version.cuda} toolkit with CUDA_HOME "
            "to match this PyTorch build"
        )

    kernel_dir = Path(__file__).resolve().parent

    load(
        name="tinyvllm_paged_decode",
        sources=[
            str(kernel_dir / "paged_decode.cu"),
            str(kernel_dir / "kv_store.cu"),
            str(kernel_dir / "pointwise.cu"),
        ],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        is_python_module=False,
    )