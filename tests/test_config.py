from __future__ import annotations

import pytest

from tinyvllm import EngineConfig


def model_dir(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    return path


def test_engine_config_validates_reservation_types(tmp_path) -> None:
    with pytest.raises(TypeError):
        EngineConfig(
            model=model_dir(tmp_path),
            prefill_reserve_tokens=True,
        )


def test_engine_config_reports_model_path_errors(tmp_path) -> None:
    with pytest.raises(ValueError, match="model directory"):
        EngineConfig(model=tmp_path / "missing")
