from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinyvllm.models import Qwen3Config


def base_config() -> dict[str, object]:
    return {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "rope_theta": 1_000_000,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": True,
        "rope_scaling": None,
        "use_sliding_window": False,
    }


def write_config(
    tmp_path: Path,
    data: dict[str, object] | None = None,
) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    payload = base_config() if data is None else data
    (model_dir / "config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return model_dir


def test_loads_qwen3_config_and_derives_projection_sizes(
    tmp_path: Path,
) -> None:
    config = Qwen3Config.from_model_dir(
        write_config(tmp_path)
    )

    assert config.hidden_size == 1024
    assert config.num_key_value_groups == 2
    assert config.q_proj_size == 16 * 128
    assert config.kv_proj_size == 8 * 128

    # This is intentionally true for Qwen3-0.6B.
    assert config.hidden_size != config.q_proj_size


def test_applies_only_safe_optional_defaults(
    tmp_path: Path,
) -> None:
    data = base_config()

    for name in (
        "rms_norm_eps",
        "hidden_act",
        "attention_bias",
        "tie_word_embeddings",
        "rope_scaling",
        "use_sliding_window",
    ):
        data.pop(name)

    config = Qwen3Config.from_model_dir(
        write_config(tmp_path, data)
    )

    assert config.rms_norm_eps == 1e-6
    assert config.hidden_act == "silu"
    assert config.attention_bias is False
    assert config.tie_word_embeddings is False


def test_requires_explicit_head_dim(
    tmp_path: Path,
) -> None:
    data = base_config()
    data.pop("head_dim")

    with pytest.raises(ValueError, match="head_dim"):
        Qwen3Config.from_model_dir(
            write_config(tmp_path, data)
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_type", "qwen3_moe", "model_type"),
        ("num_key_value_heads", 3, "divisible"),
        ("head_dim", 127, "even"),
        ("rope_theta", 0, "rope_theta"),
        ("use_sliding_window", True, "sliding-window"),
        (
            "rope_scaling",
            {"type": "linear"},
            "rope_scaling",
        ),
    ],
)
def test_rejects_unsupported_or_inconsistent_configs(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    data = base_config()
    data[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        Qwen3Config.from_model_dir(
            write_config(tmp_path, data)
        )


def test_rejects_non_causal_lm_architecture(
    tmp_path: Path,
) -> None:
    data = base_config()
    data["architectures"] = [
        "Qwen3ForSequenceClassification"
    ]

    with pytest.raises(ValueError, match="Qwen3ForCausalLM"):
        Qwen3Config.from_model_dir(
            write_config(tmp_path, data)
        )


def test_reports_missing_file_and_invalid_json(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="exist"):
        Qwen3Config.from_model_dir(
            tmp_path / "missing"
        )

    model_dir = tmp_path / "broken"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        "{broken",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        Qwen3Config.from_model_dir(model_dir)