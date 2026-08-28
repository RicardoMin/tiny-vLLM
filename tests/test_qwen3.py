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

from tinyvllm.contiguous_kv import ContiguousKVCache
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


def test_qwen3_matches_hf_with_incremental_cache(
    tmp_path,
) -> None:
    torch.manual_seed(42)

    hf_config = HfQwen3Config(
        vocab_size=41,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=6,
        max_position_embeddings=32,
        rope_theta=1_000_000,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )

    hf_model = HfQwen3ForCausalLM(
        hf_config
    ).eval()

    hf_model.save_pretrained(
        tmp_path,
        safe_serialization=True,
    )

    tiny_model = Qwen3ForCausalLM.from_pretrained(
        tmp_path,
        dtype=torch.float32,
    )

    input_ids = torch.tensor(
        [1, 4, 7, 3, 9],
        dtype=torch.long,
    )

    with torch.inference_mode():
        hf_logits = hf_model(
            input_ids[None],
            return_dict=True,
        ).logits[0]

        tiny_hidden = tiny_model(input_ids)
        tiny_logits = tiny_model.compute_logits(
            tiny_hidden
        )

        assert torch.allclose(
            tiny_logits,
            hf_logits,
            atol=1e-5,
            rtol=1e-5,
        )

        cache = ContiguousKVCache(
            num_layers=hf_config.num_hidden_layers,
            max_seq_len=32,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            device="cpu",
            dtype=torch.float32,
        )

        tiny_model(
            input_ids[:3],
            cache=cache,
        )
        assert cache.length == 3

        chunk_hidden = tiny_model(
            input_ids[3:],
            cache=cache,
        )
        assert cache.length == 5

        assert torch.allclose(
            tiny_model.compute_logits(
                chunk_hidden[-1:]
            ),
            hf_logits[-1:],
            atol=1e-5,
            rtol=1e-5,
        )

        sequence = input_ids.clone()

        for token in (11, 12, 13):
            token_tensor = torch.tensor(
                [token],
                dtype=torch.long,
            )

            decode_hidden = tiny_model(
                token_tensor,
                cache=cache,
            )

            sequence = torch.cat(
                (sequence, token_tensor)
            )

            expected = hf_model(
                sequence[None],
                return_dict=True,
            ).logits[0, -1:]

            actual = tiny_model.compute_logits(
                decode_hidden
            )

            assert torch.allclose(
                actual,
                expected,
                atol=1e-5,
                rtol=1e-5,
            )
            assert cache.length == sequence.numel()

        cache.clear()
        assert cache.length == 0