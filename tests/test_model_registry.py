from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tinyvllm.models import MODEL_REGISTRY
from tinyvllm.models.registry import ModelRegistry


def write_config(
    tmp_path: Path,
    model_type: str,
) -> Path:
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text(
        json.dumps({'model_type': model_type}),
        encoding='utf-8',
    )
    return model_dir


def test_registry_loads_factory_selected_by_model_type(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry()
    model = nn.Identity()
    calls: list[tuple[Path, object, object]] = []

    def factory(
        model_dir: str | Path,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Module:
        calls.append((Path(model_dir), device, dtype))
        return model

    registry.register('toy', factory)
    model_dir = write_config(tmp_path, 'toy')

    loaded = registry.load(
        model_dir,
        device='cpu',
        dtype=torch.float32,
    )

    assert loaded is model
    assert calls == [
        (model_dir.resolve(), 'cpu', torch.float32),
    ]


def test_registry_rejects_duplicate_and_unknown_model_types() -> None:
    registry = ModelRegistry()
    factory = lambda *args, **kwargs: nn.Identity()

    registry.register('toy', factory)

    with pytest.raises(ValueError, match='already registered'):
        registry.register('toy', factory)

    with pytest.raises(ValueError, match='unsupported'):
        registry.resolve('missing')


def test_registry_rejects_missing_model_type(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text(
        '{}',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='model_type'):
        ModelRegistry().model_type_for(model_dir)


def test_builtin_registry_contains_qwen3() -> None:
    assert callable(MODEL_REGISTRY.resolve('qwen3'))