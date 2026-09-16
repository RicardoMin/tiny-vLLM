from __future__ import annotations

import pytest

from tinyvllm.config import EngineConfig


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


@pytest.mark.parametrize("field", ["kv_block_size", "kv_cache_memory_bytes"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_engine_config_rejects_invalid_kv_geometry(tmp_path, field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        EngineConfig(model=tmp_path, **{field: value})


@pytest.mark.parametrize("value", [True, "0.9", float("nan"), float("inf"), 0, 1])
def test_engine_config_rejects_invalid_memory_utilization(tmp_path, value):
    with pytest.raises((TypeError, ValueError), match="utilization"):
        EngineConfig(model=tmp_path, gpu_memory_utilization=value)
