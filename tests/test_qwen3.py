from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyvllm.models import Qwen3Config

import torch
from transformers import (
    Qwen3Config as HfQwen3Config,
    Qwen3ForCausalLM as HfQwen3ForCausalLM,
)

from tinyvllm.models.qwen3 import Qwen3ForCausalLM


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


def test_reads_qwen3_geometry(
    tmp_path: Path,
) -> None:
    config = Qwen3Config.from_model_dir(
        write_config(tmp_path)
    )

    assert config.hidden_size == 1024
    assert config.num_key_value_groups == 2
    assert config.q_proj_size == 2048
    assert config.kv_proj_size == 1024
    assert config.qkv_proj_size == 4096


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


def test_rejects_incomplete_source_config() -> None:
    with pytest.raises(ValueError, match="missing Qwen3"):
        Qwen3Config.from_dict({"model_type": "qwen3"})


@pytest.mark.parametrize("name", [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.0.self_attn.o_proj.weight",
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
