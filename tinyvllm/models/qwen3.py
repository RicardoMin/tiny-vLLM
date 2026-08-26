from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinyvllm._validation import _check_positive_int


_REQUIRED_INT_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
)


def _positive_float(name: str, value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a number")

    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")

    return value


def _required(data: Mapping[str, Any], name: str) -> Any:
    if name not in data:
        raise ValueError(
            f"missing required Qwen3 config field: {name}"
        )
    return data[name]


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    """Validated static structure of a dense Qwen3 checkpoint."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float

    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention_bias: bool = False
    tie_word_embeddings: bool = False


    def __post_init__(self) -> None:
        for name in _REQUIRED_INT_FIELDS:
            _check_positive_int(name, getattr(self, name))

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads"
            )

        # Rotary embedding consumes pairs of values.
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        rms_norm_eps = _positive_float(
            "rms_norm_eps",
            self.rms_norm_eps,
        )
        rope_theta = _positive_float(
            "rope_theta",
            self.rope_theta,
        )

        object.__setattr__(self, "rms_norm_eps", rms_norm_eps)
        object.__setattr__(self, "rope_theta", rope_theta)

        if type(self.hidden_act) is not str:
            raise TypeError("hidden_act must be a string")
        if self.hidden_act != "silu":
            raise ValueError(
                "tinyvllm v1 only supports hidden_act='silu'"
            )

        if type(self.attention_bias) is not bool:
            raise TypeError("tie_word_embeddings must be bool")
        if self.attention_bias:
            raise ValueError(
                "tinyvllm v1 requires attention_bias=False"
            )

        if type(self.tie_word_embeddings) is not bool:
            raise TypeError(
                "tie_word_embeddings must be bool"
            )


    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_proj_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "Qwen3Config":
        if not isinstance(data, Mapping):
            raise TypeError(
                "Qwen3 config must be a JSON object"
            )

        if data.get("model_type") != "qwen3":
            raise ValueError(
                "only dense Qwen3 checkpoints "
                "(model_type='qwen3') are supported"
            )

        architectures = data.get("architectures")
        if architectures is not None:
            if isinstance(architectures, str):
                names = (architectures,)
            elif isinstance(architectures, list):
                names = tuple(architectures)
            else:
                raise TypeError(
                    "architectures must be a string or list"
                )

            if not all(type(name) is str for name in names):
                raise TypeError(
                    "architectures entries must be strings"
                )

            if "Qwen3ForCausalLM" not in names:
                raise ValueError(
                    "checkpoint is not Qwen3ForCausalLM"
                )

        # We do not silently ignore features that the v1 model does not implement.
        if data.get("rope_scaling") is not None:
            raise ValueError(
                "rope_scaling is not supported by tinyvllm v1"
            )

        use_sliding_window = data.get(
            "use_sliding_window",
            False,
        )
        if type(use_sliding_window) is not bool:
            raise TypeError(
                "use_sliding_window must be bool"
            )
        if use_sliding_window:
            raise ValueError(
                "sliding-window attention is not supported by "
                "tinyvllm v1"
            )

        values = {
            name: _required(data, name)
            for name in _REQUIRED_INT_FIELDS
        }
        values["rope_theta"] = _required(
            data,
            "rope_theta",
        )

        values.update(
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            hidden_act=data.get("hidden_act", "silu"),
            attention_bias=data.get("attention_bias", False),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
        )

        return cls(**values)


    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
    ) -> "Qwen3Config":
        model_dir = Path(model_dir).expanduser().resolve()

        if not model_dir.is_dir():
            raise ValueError(
                f"model directory does not exist: {model_dir}"
            )

        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise ValueError(
                f"missing config.json: {config_path}"
            )

        try:
            with config_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON in {config_path}"
            ) from exc

        return cls.from_dict(data)