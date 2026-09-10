from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import torch
from torch import nn

from tinyvllm._validation import _require_model_type


class ModelFactory(Protocol):
    def __call__(
        self,
        model_dir: str | Path,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None  = None,
    ) -> nn.Module: ...




class ModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}


    def register(
        self,
        model_type: str,
        factory: ModelFactory,
    ) -> None:
        model_type = _require_model_type(model_type)

        if not callable(factory):
            raise TypeError("model factory must be callable")
        if model_type in self._factories:
            raise ValueError(
                f"model_type already registered: {model_type}"
            )

        self._factories[model_type] = factory


    def resolve(self, model_type: str) -> ModelFactory:
        model_type = _require_model_type(model_type)

        try:
            return self._factories[model_type]
        except KeyError as exc:
            raise ValueError(
                f"unsupported model_type: {model_type}"
            ) from exc


    def model_type_for(
        self,
        model_dir: str | Path,
    ) -> str:
        root = Path(model_dir).expanduser().resolve()
        config_path = root / "config.json"

        try:
            source = json.loads(
                config_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read model config: {config_path}"
            ) from exc

        if not isinstance(source, dict):
            raise ValueError("model config must be a JSON object")

        return _require_model_type(source.get("model_type"))


    def load(
        self,
        model_dir: str | Path,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Module:
        root = Path(model_dir).expanduser().resolve()
        factory = self.resolve(self.model_type_for(root))
        return factory(root, device=device, dtype=dtype)