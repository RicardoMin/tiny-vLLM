from __future__ import annotations

from collections.abc import Sequence

import torch

from tinyvllm.request import SamplingParams


class TorchSampler:
    """Correctness-first eager sampler with per-request RNG streams."""

    def __init__(self) -> None:
        self._generators: dict[
            int,
            tuple[int, torch.device, torch.Generator],
        ] = {}


    @torch.inference_mode()
    def sample(
        self,
        logits: torch.Tensor,
        sampling_params: Sequence[SamplingParams],
        request_ids: Sequence[int],
    ) -> torch.Tensor:
        num_rows, vocab_size = logits.shape
        if len(sampling_params) != num_rows:
            raise ValueError(
                "sampling_params must align with logits rows"
            )
        if len(request_ids) != num_rows:
            raise ValueError("request_ids must align with logits rows")

        if num_rows == 0:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=logits.device,
            )

        if all(params.greedy for params in sampling_params):
            return logits.argmax(dim=-1)

        sampled_rows: list[torch.Tensor] = []

        for row, params, request_id in zip(
            logits,
            sampling_params,
            request_ids,
        ):
            if params.greedy:
                token_id = row.argmax(dim=0, keepdim=True)
            else:
                token_id = self._sample_row(
                    row,
                    params,
                    request_id,
                )

            sampled_rows.append(token_id)

        return torch.cat(sampled_rows, dim=0)


    def _sample_row(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        request_id: int,
    ) -> torch.Tensor:
        scores = logits.float().div(params.temperature)
        candidate_ids: torch.Tensor | None = None

        if 0 < params.top_k < scores.numel():
            scores, candidate_ids = torch.topk(
                scores,
                k=params.top_k,
            )

        if scores.numel() == 1:
            if candidate_ids is None:
                return torch.zeros(
                    (1,),
                    dtype=torch.long,
                    device=torch.device,
                )
            return candidate_ids[:1]

        generator = self._generator(
            request_id,
            params.seed,
            logits.device,
        )

        if params.top_p < 1.0:
            sorted_scores, order = torch.sort(
                scores,
                descending=True,
                stable=True,
            )
            probabilities = torch.softmax(
                sorted_scores,
                dim=0,
            )

            remove = probabilities.cumsum(dim=0) > params.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            probabilities = probabilities.masked_fill(
                remove,
                0.0,
            )

            choice = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
            choice = order.gather(0, choice)
        else:
            probabilities = torch.softmax(scores, dim=0)
            choice = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )

        if candidate_ids is not None:
            choice = candidate_ids.gather(0, choice)

        return choice


    def _generator(
        self,
        request_id: int,
        seed: int,
        device: torch.device,
    ) -> torch.Generator:
        state = self._generators.get(request_id)

        if (
            state is None
            or state[0] != seed
            or state[1] != device
        ):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            state = (seed, device, generator)
            self._generators[request_id] = state

        return state[2]


    def release(self, request_id: int) -> None:
        self._generators.pop(request_id, None)


    def clear(self) -> None:
        self._generators.clear()