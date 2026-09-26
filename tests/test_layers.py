from __future__ import annotations

import pytest
import torch

from tinyvllm.layers import (
    PackedLinear,
    RMSNorm,
    RotaryEmbedding,
)


def test_packed_linear_loads_named_row_ranges() -> None:
    layer = PackedLinear(
        input_size=3,
        output_sizes=(2, 4, 1),
        shard_names=("q", "k", "v"),
        dtype=torch.float32,
    )

    with torch.no_grad():
        layer.weight.zero_()

    layer.load_shard(
        "q",
        torch.full((2, 3), 1.0),
    )
    layer.load_shard(
        "k",
        torch.full((4, 3), 2.0),
    )
    layer.load_shard(
        "v",
        torch.full((1, 3), 3.0),
    )

    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ]
    )

    assert torch.equal(
        layer.weight,
        expected,
    )

    assert torch.equal(
        layer(torch.ones(1, 3)),
        torch.tensor([[3.0, 3.0, 6.0, 6.0, 6.0, 6.0, 9.0]]),
    )


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_rmsnorm_matches_hf_rounding_with_nonuniform_weights(dtype):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    from tinyvllm.layers import RMSNorm
    torch.manual_seed(17)
    actual = RMSNorm(64, dtype=dtype)
    oracle = Qwen3RMSNorm(64).to(dtype)
    with torch.no_grad():
        actual.weight.copy_(torch.randn(64))
        oracle.weight.copy_(actual.weight)
    x = torch.randn(3, 5, 64).to(dtype)
    torch.testing.assert_close(actual(x), oracle(x), atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_rope_matches_hf_rounding_at_nonzero_positions(dtype):
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb
    from tinyvllm.layers import RotaryEmbedding
    torch.manual_seed(19)
    config = Qwen3Config(hidden_size=128, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=32,
                        max_position_embeddings=4096, rope_theta=1_000_000)
    actual = RotaryEmbedding(32, 4096)
    oracle = Qwen3RotaryEmbedding(config)
    positions = torch.tensor([0, 37, 1023, 4095])
    q, k = torch.randn(4, 4, 32).to(dtype), torch.randn(4, 2, 32).to(dtype)
    cos, sin = oracle(q, positions[None])
    expected_q, expected_k = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin)
    actual_q, actual_k = actual(positions, q, k)
    torch.testing.assert_close(actual_q, expected_q[0].transpose(0, 1), atol=0, rtol=0)
    torch.testing.assert_close(actual_k, expected_k[0].transpose(0, 1), atol=0, rtol=0)
