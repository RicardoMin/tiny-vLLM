from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from tinyvllm._validation import _check_positive_int
from tinyvllm.checkpoint import load_weights
from tinyvllm.runtime.contracts import AttentionBackend
from tinyvllm.layers import (
    PackedLinear,
    RMSNorm,
    RotaryEmbedding,
    SiluAndMul,
)


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
    _hf_config: Any

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int
    num_attention_heads: int
    num_key_value_heads: int
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

        dimension_names = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "max_position_embeddings",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
        try:
            dimensions = tuple(getattr(config, name) for name in dimension_names)
        except AttributeError as exc:
            raise ValueError("missing Qwen3 dimensions") from exc

        for name, value in zip(dimension_names, dimensions):
            _check_positive_int(name, value)

        (
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            max_position_embeddings,
            heads,
            kv_heads,
            head_dim,
        ) = dimensions

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

        return cls(
            _hf_config=config,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hf_config, name)

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


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        rotary_emb: RotaryEmbedding,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.rotary_emb = rotary_emb

        self.qkv_proj = PackedLinear(
            config.hidden_size,
            (
                config.q_proj_size,
                config.kv_proj_size,
                config.kv_proj_size,
            ),
            shard_names=("q", "k", "v"),
            device=device,
            dtype=dtype,
        )

        self.o_proj = nn.Linear(
            config.q_proj_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.q_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        layer_idx: int,
        attention: AttentionBackend,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        q, k, v = self.qkv_proj(hidden_states).split(
            (
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ),
            dim=-1,
        )

        q = self.q_norm(
            q.view(
                num_tokens,
                self.num_heads,
                self.head_dim,
            )
        )
        k = self.k_norm(
            k.view(
                num_tokens,
                self.num_kv_heads,
                self.head_dim,
            )
        )
        v = v.view(
            num_tokens,
            self.num_kv_heads,
            self.head_dim,
        )

        q, k = self.rotary_emb(
            positions,
            q,
            k,
        )

        output = attention(layer_idx, q, k, v, scale=self.scale)

        return self.o_proj(output.reshape(num_tokens, -1))


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()

        self.gate_up_proj = PackedLinear(
            config.hidden_size,
            (
                config.intermediate_size,
                config.intermediate_size,
            ),
            shard_names=("gate", "up"),
            device=device,
            dtype=dtype,
        )

        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.act_fn = SiluAndMul()


    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(
                self.gate_up_proj(hidden_states)
            )
        )


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        rotary_emb: RotaryEmbedding,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.self_attn = Qwen3Attention(
            config,
            rotary_emb,
            device=device,
            dtype=dtype,
        )

        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )

        self.mlp = Qwen3MLP(
            config,
            device=device,
            dtype=dtype,
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        layer_idx: int,
        attention: AttentionBackend,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(
            hidden_states
        )
        hidden_states = residual + self.self_attn(
            hidden_states,
            positions,
            layer_idx=layer_idx,
            attention=attention,
        )

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(
            hidden_states
        )
        hidden_states = residual + self.mlp(
            hidden_states
        )

        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )

        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            base=config.rope_theta,
            device=device,
        )

        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(
                config,
                self.rotary_emb,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.num_hidden_layers)
        )

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        attention: AttentionBackend,
    ) -> torch.Tensor:
        if input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError(
                "input_ids must be a flat 1-D tensor"
            )

        if input_ids.dtype is not torch.long or positions.dtype is not torch.long:
            raise TypeError(
                "input_ids and positions must use torch.long"
            )

        if positions.shape != input_ids.shape or positions.device != input_ids.device:
            raise ValueError(
                "positions must align with input_ids on the same device"
            )

        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, positions, layer_idx=layer_idx, attention=attention,
            )

        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.model = Qwen3Model(
            config,
            device=device,
            dtype=dtype,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device="meta" if config.tie_word_embeddings else device,
            dtype=dtype,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = (
                self.model.embed_tokens.weight
            )

        self._weight_routes = (
            self._make_weight_routes()
        )


    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Qwen3ForCausalLM":
        config = Qwen3Config.from_model_dir(
            model_dir
        )

        model = cls(
            config,
            device=device,
            dtype=dtype,
        )

        load_weights(
            model,
            model_dir,
        )

        model.eval()
        return model


    @property
    def required_weight_names(self) -> frozenset[str]:
        names = set(self._weight_routes)

        if self.config.tie_word_embeddings:
            names.remove("lm_head.weight")

        return frozenset(names)


    def _make_weight_routes(
        self,
    ) -> dict[str, tuple[str, str | None]]:
        routes: dict[str, tuple[str, str | None]] = {
            "model.embed_tokens.weight": (
                "model.embed_tokens.weight",
                None,
            ),
            "model.norm.weight": (
                "model.norm.weight",
                None,
            ),
        }

        for layer_idx in range(
            self.config.num_hidden_layers
        ):
            prefix = f"model.layers.{layer_idx}"

            routes.update(
                {
                    f"{prefix}.input_layernorm.weight": (
                        f"{prefix}.input_layernorm.weight",
                        None,
                    ),
                    f"{prefix}.post_attention_layernorm.weight": (
                        f"{prefix}.post_attention_layernorm.weight",
                        None,
                    ),
                    f"{prefix}.self_attn.q_proj.weight": (
                        f"{prefix}.self_attn.qkv_proj",
                        "q",
                    ),
                    f"{prefix}.self_attn.k_proj.weight": (
                        f"{prefix}.self_attn.qkv_proj",
                        "k",
                    ),
                    f"{prefix}.self_attn.v_proj.weight": (
                        f"{prefix}.self_attn.qkv_proj",
                        "v",
                    ),
                    f"{prefix}.self_attn.o_proj.weight": (
                        f"{prefix}.self_attn.o_proj.weight",
                        None,
                    ),
                    f"{prefix}.self_attn.q_norm.weight": (
                        f"{prefix}.self_attn.q_norm.weight",
                        None,
                    ),
                    f"{prefix}.self_attn.k_norm.weight": (
                        f"{prefix}.self_attn.k_norm.weight",
                        None,
                    ),
                    f"{prefix}.mlp.gate_proj.weight": (
                        f"{prefix}.mlp.gate_up_proj",
                        "gate",
                    ),
                    f"{prefix}.mlp.up_proj.weight": (
                        f"{prefix}.mlp.gate_up_proj",
                        "up",
                    ),
                    f"{prefix}.mlp.down_proj.weight": (
                        f"{prefix}.mlp.down_proj.weight",
                        None,
                    ),
                }
            )

        if self.config.tie_word_embeddings:
            routes["lm_head.weight"] = (
                "model.embed_tokens.weight",
                None,
            )
        else:
            routes["lm_head.weight"] = (
                "lm_head.weight",
                None,
            )

        return routes


    def load_weight(
        self,
        source_name: str,
        tensor: torch.Tensor,
    ) -> str:
        try:
            target_name, shard = (
                self._weight_routes[source_name]
            )
        except KeyError as exc:
            raise KeyError(
                f"unexpected Qwen3 weight: {source_name}"
            ) from exc

        if shard is None:
            target = self.get_parameter(target_name)
            if tensor.shape != target.shape:
                raise ValueError(
                    f"{source_name}: expected shape {tuple(target.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            with torch.no_grad():
                target.copy_(tensor)
        else:
            self.get_submodule(
                target_name
            ).load_shard(
                shard,
                tensor,
            )

        if (
            self.config.tie_word_embeddings
            and source_name == "lm_head.weight"
        ):
            return "model.embed_tokens.weight"

        return source_name

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        attention: AttentionBackend,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            attention=attention,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_indices is not None:
            hidden_states = hidden_states.index_select(
                0,
                token_indices,
            )

        return F.linear(
            hidden_states,
            self.lm_head.weight,
        )
