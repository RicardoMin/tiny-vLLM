from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    hf_config: Any

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

        config = AutoConfig.from_pretrained(
            str(root),
            local_files_only=True,
            trust_remote_code=False,
        )

        return cls.from_hf(config)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "Qwen3Config":
        from transformers import Qwen3Config as HfQwen3Config

        return cls.from_hf(
            HfQwen3Config.from_dict(data)
        )

    @classmethod
    def from_hf(
        cls,
        config: Any,
    ) -> "Qwen3Config":
        if config.model_type != "qwen3":
            raise ValueError(
                "only model_type='qwen3' is supported"
            )

        heads = int(config.num_attention_heads)
        kv_heads = int(
            getattr(
                config,
                "num_key_value_heads",
                heads,
            )
        )

        if heads <= 0 or kv_heads <= 0:
            raise ValueError(
                "attention head counts must be positive"
            )

        if heads % kv_heads != 0:
            raise ValueError(
                "attention heads must be divisible by "
                "key-value heads"
            )

        raw_head_dim = getattr(
            config,
            "head_dim",
            None,
        )

        head_dim = (
            int(raw_head_dim)
            if raw_head_dim is not None
            else int(config.hidden_size) // heads
        )

        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be a positive even number"
            )

        if getattr(config, "rope_scaling", None):
            raise ValueError(
                "rope_scaling is not supported in v1"
            )

        if getattr(
            config,
            "use_sliding_window",
            False,
        ):
            raise ValueError(
                "sliding-window attention is not supported"
            )

        return cls(config)

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