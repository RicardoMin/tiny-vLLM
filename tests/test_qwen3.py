from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyvllm.models import Qwen3Config


def write_config(tmp_path: Path) -> Path:
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


def test_rejects_unsupported_attention_features(
    tmp_path: Path,
) -> None:
    model_dir = write_config(tmp_path)
    data = json.loads(
        (model_dir / "config.json").read_text()
    )
    data["rope_scaling"] = {"factor": 2.0}

    (model_dir / "config.json").write_text(
        json.dumps(data),
        encoding="utf-8",
    )
