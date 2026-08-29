from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from tinyvllm.models.qwen3 import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
)
from tinyvllm.models.registry import ModelRegistry


MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.register(
    'qwen3',
    Qwen3ForCausalLM.from_pretrained,
)


def load_model(
    model_dir: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    return MODEL_REGISTRY.load(
        model_dir,
        device=device,
        dtype=dtype,
    )


__all__ = [
    'MODEL_REGISTRY',
    'ModelRegistry',
    'Qwen3Config',
    'Qwen3ForCausalLM',
    'Qwen3Model',
    'load_model',
]