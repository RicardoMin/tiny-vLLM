from __future__ import annotations

import pytest
import torch

from tinyvllm.request import SamplingParams
from tinyvllm.sampling import TorchSampler


def stochastic_params(
    *,
    seed: int = 0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> SamplingParams:
    return SamplingParams(
        max_new_tokens=1,
        temperature=1.0,
        seed=seed,
        top_k=top_k,
        top_p=top_p,
    )


def test_empty_batch_preserves_output_contract() -> None:
    sampled = TorchSampler().sample(
        torch.empty((0, 4), dtype=torch.float32),
        (),
        (),
    )

    assert sampled.shape == (0,)
    assert sampled.dtype == torch.long
    assert sampled.device.type == "cpu"


def test_greedy_ignores_top_k_and_top_p() -> None:
    logits = torch.tensor(
        [
            [1.0, 7.0, 3.0],
            [9.0, 2.0, 4.0],
        ]
    )
    params = SamplingParams(
        temperature=0.0,
        top_k=1,
        top_p=0.01,
    )

    sampled = TorchSampler().sample(
        logits,
        (params, params),
        (11, 22),
    )

    assert sampled.tolist() == [1, 0]


def test_top_k_one_is_deterministic() -> None:
    sampled = TorchSampler().sample(
        torch.tensor([[1.0, 8.0, 4.0]]),
        (stochastic_params(seed=7, top_k=1),),
        (11,),
    )

    assert sampled.tolist() == [1]


def test_top_p_keeps_token_that_crosses_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def choose_last_positive(
        weights: torch.Tensor,
        num_samples: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        assert num_samples == 1
        positive = torch.nonzero(
            weights > 0,
            as_tuple=False,
        ).flatten()
        return positive[-1:].to(weights.device)

    monkeypatch.setattr(
        torch,
        "multinomial",
        choose_last_positive,
    )

    logits = torch.tensor([[0.50, 0.30, 0.20]]).log()
    sampled = TorchSampler().sample(
        logits,
        (stochastic_params(top_p=0.60),),
        (11,),
    )

    assert sampled.tolist() == [1]


def test_request_rng_is_independent_of_batch_order() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [1.0, 0.0, -1.0, 0.5],
        ]
    )
    first_params = stochastic_params(seed=17)
    second_params = stochastic_params(seed=29)

    forward = TorchSampler()
    reversed_order = TorchSampler()

    for _ in range(12):
        forward_tokens = forward.sample(
            logits,
            (first_params, second_params),
            (101, 202),
        )
        reversed_tokens = reversed_order.sample(
            logits.flip(0),
            (second_params, first_params),
            (202, 101),
        )

        assert torch.equal(
            forward_tokens,
            reversed_tokens.flip(0),
        )


def test_release_restarts_request_rng_stream() -> None:
    sampler = TorchSampler()
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    params = stochastic_params(seed=123)

    def draw_sequence() -> torch.Tensor:
        return torch.cat(
            [
                sampler.sample(
                    logits,
                    (params,),
                    (7,),
                )
                for _ in range(12)
            ]
        )

    first = draw_sequence()
    sampler.release(7)
    replayed = draw_sequence()

    assert torch.equal(first, replayed)


def test_sampler_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="sampling_params"):
        TorchSampler().sample(
            torch.zeros((1, 4)),
            (),
            (7,),
        )