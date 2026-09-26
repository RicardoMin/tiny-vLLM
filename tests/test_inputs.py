"""External configuration and sampling boundaries; lifecycle checks live at Engine."""

import pytest

from tinyvllm.config import EngineConfig
from tinyvllm.request import SamplingParams


def test_engine_config_validates_reservation_types(tmp_path) -> None:
    with pytest.raises(TypeError):
        EngineConfig(
            model=tmp_path,
            prefill_reserve_tokens=True,
        )


def test_engine_config_reports_model_path_errors(tmp_path) -> None:
    with pytest.raises(ValueError, match="model directory"):
        EngineConfig(model=tmp_path / "missing")


@pytest.mark.parametrize("field,value", [
    ("kv_block_size", 0), ("kv_block_size", True),
    ("kv_cache_memory_bytes", -1), ("kv_cache_memory_bytes", 1.5),
])
def test_engine_config_rejects_invalid_kv_geometry(tmp_path, field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        EngineConfig(model=tmp_path, **{field: value})


@pytest.mark.parametrize("value", [True, "0.9", float("nan"), float("inf"), 0, 1])
def test_engine_config_rejects_invalid_memory_utilization(tmp_path, value):
    with pytest.raises((TypeError, ValueError), match="utilization"):
        EngineConfig(model=tmp_path, gpu_memory_utilization=value)


def test_attention_backend_is_explicit_and_reference_is_default(tmp_path):
    assert EngineConfig(model=tmp_path).attention_backend == "reference"
    for backend in ("reference", "cuda", "flashinfer"):
        assert EngineConfig(model=tmp_path, attention_backend=backend).attention_backend == backend
    for invalid in ("auto", "automatic"):
        with pytest.raises(ValueError, match="attention backend"):
            EngineConfig(model=tmp_path, attention_backend=invalid)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_new_tokens": -1}, ValueError),
        ({"max_new_tokens": True}, TypeError),
        ({"temperature": float("nan")}, ValueError),
        ({"temperature": True}, TypeError),
        ({"ignore_eos": 1}, TypeError),
        ({"seed": -1}, ValueError),
        ({"seed": 1 << 64}, ValueError),
        ({"top_k": -1}, ValueError),
        ({"top_k": True}, TypeError),
        ({"top_p": 0.0}, ValueError),
        ({"top_p": 1.01}, ValueError),
        ({"top_p": float("nan")}, ValueError),
        ({"top_p": True}, TypeError),
    ],
)
def test_sampling_rejects_invalid_values(kwargs, error) -> None:
    with pytest.raises(error):
        SamplingParams(**kwargs)
