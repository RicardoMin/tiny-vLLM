from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyvllm.models import Qwen3Config

import torch
from transformers import (
    Qwen3Config as HfQwen3Config,
)

from tinyvllm.models.qwen3 import Qwen3ForCausalLM


@pytest.mark.parametrize(
    "entry,value",
    [("from_dict", 1.9), ("from_model_dir", True), ("from_hf", "2")],
)
def test_dimension_types_are_not_silently_coerced(tmp_path, entry, value):
    model_dir = write_config(tmp_path, num_hidden_layers=value)
    data = json.loads((model_dir / "config.json").read_text())
    if entry == "from_dict":
        source = data
    elif entry == "from_model_dir":
        source = model_dir
    else:
        source = HfQwen3Config.from_dict(data)

    with pytest.raises(TypeError, match="num_hidden_layers must be an integer"):
        getattr(Qwen3Config, entry)(source)


def write_config(
    tmp_path: Path,
    **updates: object,
) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    data = {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "rope_theta": 1_000_000,
        "tie_word_embeddings": True,
    }
    data.update(updates)

    (model_dir / "config.json").write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    return model_dir


@pytest.mark.parametrize(
    "updates",
    [
        {"rope_scaling": {"factor": 2.0}},
        {"hidden_act": "gelu"},
        {"attention_bias": True},
    ],
)
def test_rejects_unsupported_model_features(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        Qwen3Config.from_model_dir(
            write_config(tmp_path, **updates)
        )


@pytest.mark.parametrize("tied", [False, True])
def test_lm_head_binding_avoids_temporary_storage_and_loads_weights(tied, monkeypatch):
    config = Qwen3Config.from_hf(HfQwen3Config(
        vocab_size=41, hidden_size=24, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=6,
        max_position_embeddings=32, tie_word_embeddings=tied,
    ))
    initialization_devices = []
    original = torch.nn.Linear.reset_parameters

    def record(layer):
        if (layer.out_features, layer.in_features) == (41, 24):
            initialization_devices.append(layer.weight.device.type)
        original(layer)

    monkeypatch.setattr(torch.nn.Linear, "reset_parameters", record)
    model = Qwen3ForCausalLM(config, device="cpu", dtype=torch.float32)
    embedding = model.model.embed_tokens.weight
    assert initialization_devices == ["meta" if tied else "cpu"]
    assert (model.lm_head.weight is embedding) == tied
    assert all(not tensor.is_meta for tensor in model.state_dict().values())

    embedding_before = embedding.detach().clone()
    weights = torch.full_like(model.lm_head.weight, 0.25)
    model.load_weight("lm_head.weight", weights)
    torch.testing.assert_close(model.lm_head.weight, weights)
    torch.testing.assert_close(embedding, weights if tied else embedding_before)
    hidden = torch.randn(3, config.hidden_size)
    torch.testing.assert_close(
        model.compute_logits(hidden),
        torch.nn.functional.linear(hidden, weights),
    )


def test_rejects_incomplete_source_config() -> None:
    with pytest.raises(ValueError, match="missing Qwen3"):
        Qwen3Config.from_dict({"model_type": "qwen3"})


@pytest.mark.parametrize("name", [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
])
def test_rejects_broadcastable_weight_shape_before_copy(name):
    config = Qwen3Config.from_hf(HfQwen3Config(
        vocab_size=41, hidden_size=24, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=6,
        max_position_embeddings=32, tie_word_embeddings=True,
    ))
    model = Qwen3ForCausalLM(config)
    target = model.get_parameter(name)
    before = target.detach().clone()
    malformed = torch.zeros((1,) + tuple(target.shape[1:]))
    with pytest.raises(ValueError, match="shape"):
        model.load_weight(name, malformed)
    torch.testing.assert_close(target, before)
    correct = torch.ones_like(target, dtype=torch.float64)
    model.load_weight(name, correct)
    torch.testing.assert_close(target, correct.to(target.dtype))
