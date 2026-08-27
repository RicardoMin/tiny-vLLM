from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REQUIRED_FIELDS = frozenset(
    {
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
    }
)


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    hf_config: Any
    head_dim: int

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
    ) -> "Qwen3Config":
        from transformers import AutoConfig

        root = Path(model_dir).expanduser().resolve()

        if not root.is_dir():
            raise ValueError(
                f"model directory does not exist: {root}"
            )

        config_path = root / "config.json"

        try:
            source = json.loads(
                config_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read Qwen3 config: {config_path}"
            ) from exc

        cls._validate_source(source)

        config = AutoConfig.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
        )

        return cls.from_hf(config)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "Qwen3Config":
        from transformers import Qwen3Config as HfQwen3Config

        cls._validate_source(data)

        return cls.from_hf(
            HfQwen3Config.from_dict(dict(data))
        )

    @staticmethod
    def _validate_source(data: object) -> None:
        if not isinstance(data, Mapping):
            raise ValueError("Qwen3 config must be a JSON object")

        if data.get("model_type") != "qwen3":
            raise ValueError(
                "only model_type='qwen3' is supported"
            )

        missing = _REQUIRED_FIELDS - data.keys()

        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"missing Qwen3 config fields: {names}"
            )

        if data.get("rope_scaling"):
            raise ValueError(
                "rope_scaling is not supported in v1"
            )

    @classmethod
    def from_hf(
        cls,
        config: Any,
    ) -> "Qwen3Config":
        if getattr(config, "model_type", None) != "qwen3":
            raise ValueError(
                "only model_type='qwen3' is supported"
            )

        try:
            dimensions = (
                int(config.vocab_size),
                int(config.hidden_size),
                int(config.intermediate_size),
                int(config.num_hidden_layers),
                int(config.max_position_embeddings),
                int(config.num_attention_heads),
                int(config.num_key_value_heads),
                int(config.head_dim),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("invalid Qwen3 dimensions") from exc

        (
            _,
            _,
            _,
            _,
            _,
            heads,
            kv_heads,
            head_dim,
        ) = dimensions

        if min(dimensions) <= 0:
            raise ValueError(
                "Qwen3 dimensions must be positive"
            )

        if heads % kv_heads != 0:
            raise ValueError(
                "attention heads must be divisible by "
                "key-value heads"
            )

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        if getattr(config, "hidden_act", "silu") != "silu":
            raise ValueError(
                "only hidden_act='silu' is supported"
            )

        if getattr(config, "attention_bias", False):
            raise ValueError(
                "attention_bias is not supported"
            )

        if getattr(config, "rope_scaling", None):
            raise ValueError(
                "rope_scaling is not supported in v1"
            )

        if getattr(config, "use_sliding_window", False):
            raise ValueError(
                "sliding-window attention is not supported"
            )

        return cls(config, head_dim)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.hf_config, name)

    @property
    def num_key_value_groups(self) -> int:
        return (
            self.num_attention_heads
            // self.num_key_value_heads
        )

    @property
    def q_proj_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def qkv_proj_size(self) -> int:
        return self.q_proj_size + 2 * self.kv_proj_size
