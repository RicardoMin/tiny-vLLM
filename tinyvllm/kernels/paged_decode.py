from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import subprocess

import torch

from tinyvllm.paged_kv import PagedKVCache
from tinyvllm.runtime.contracts import SingleQueryMetadata


@lru_cache(maxsize=1)
def load_paged_decode() -> None:
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

    load(
        name="tinyvllm_paged_decode",
        sources=[str(Path(__file__).with_suffix(".cu"))],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        is_python_module=False,
    )


def paged_decode_attention(
    q: torch.Tensor,
    cache: PagedKVCache,
    layer_idx: int,
    metadata: SingleQueryMetadata,
    output: torch.Tensor,
    *,
    scale: float,
) -> None:
    """File selected query rows on PyTorch's current CUDA stream."""
    load_paged_decode()

    torch.ops.tinyvllm_cuda.paged_decode(
        q,
        cache.key[layer_idx],
        cache.value[layer_idx],
        metadata.query_rows,
        metadata.block_table_ids,
        metadata.block_table_offsets,
        metadata.context_lens,
        output,
        scale,
        cache.block_size,
    )