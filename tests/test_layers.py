from __future__ import annotations

import pytest
import torch

from tinyvllm.layers import (
    PackedLinear,
    RMSNorm,
    RotaryEmbedding,
    SiluAndMul,
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


def test_rmsnorm_matches_reference_formula() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.5, 2.0, -3.0],
        ]
    )

    layer = RMSNorm(
        hidden_size=4,
        eps=1e-6,
        dtype=torch.float32,
    )

    with torch.no_grad():
        layer.weight.fill_(1.0)

    expected = x * torch.rsqrt(
        x.square().mean(
            dim=-1,
            keepdim=True,
        )
        + 1e-6
    )

    assert torch.allclose(
        layer(x),
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_rotary_embedding_uses_half_split_rotation() -> None:
    rope = RotaryEmbedding(
        head_dim=4,
        max_position=8,
        base=10.0,
    )

    query = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            [
                [2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0],
            ],
        ]
    )
    key = torch.tensor(
        [
            [[-1.0, -2.0, -3.0, -4.0]],
            [[-2.0, -3.0, -4.0, -5.0]],
        ]
    )

    positions = torch.tensor(
        [0, 2],
        dtype=torch.long,
    )

    rotated_query, rotated_key = rope(
        positions,
        query,
        key,
    )

    inv_freq = 1.0 / (
        10.0
        ** (
            torch.arange(
                0,
                4,
                2,
                dtype=torch.float32,
            )
            / 4
        )
    )

    frequencies = positions.float().unsqueeze(-1) * inv_freq
    cos = frequencies.cos().unsqueeze(1)
    sin = frequencies.sin().unsqueeze(1)

    first, second = query.float().chunk(
        2,
        dim=-1,
    )

    expected = torch.cat(
        (
            first * cos - second * sin,
            second * cos + first * sin,
        ),
        dim=-1,
    )

    assert torch.allclose(
        rotated_query,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )

    first, second = key.float().chunk(2, dim=-1)
    expected_key = torch.cat(
        (
            first * cos - second * sin,
            second * cos + first * sin,
        ),
        dim=-1,
    )

    assert torch.allclose(
        rotated_key,
        expected_key,
        atol=1e-6,
        rtol=1e-6,
    )
    assert rope.cos.shape == (8, 2)


def test_silu_and_mul() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
        ]
    )

    result = SiluAndMul()(x)

    expected = torch.nn.functional.silu(
        x[:, :2]
    ) * x[:, 2:]

    assert torch.allclose(
        result,
        expected,
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
