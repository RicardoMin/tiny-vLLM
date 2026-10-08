from __future__ import annotations

from collections.abc import Sequence

import torch

from tinyvllm.request import SamplingParams


class TorchSampler:
    """Batched probability transforms with per-request RNG streams."""

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
        num_rows, _ = logits.shape
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

        first = sampling_params[0]
        policy = (first.temperature, first.top_k, first.top_p)
        
        if not first.greedy and all (
            (params.temperature, params.top_k, params.top_p) == policy
            for params in sampling_params
        ):
            return self._sample_batch(logits, sampling_params, request_ids)

        sampled_rows: list[torch.Tensor] = []

        for row, params, request_id in zip(
            logits,
            sampling_params,
            request_ids,
        ):
            if params.greedy:
                token_id = row.argmax(dim=0, keepdim=True)
            else:
                token_id = self._sample_batch(
                    row.unsqueeze(0),
                    (params,),
                    (request_id,),
                )

            sampled_rows.append(token_id)

        return torch.cat(sampled_rows, dim=0)


    def _sample_batch(
        self,
        logits: torch.Tensor,
        sampling_params: Sequence[SamplingParams],
        request_ids: Sequence[int],
    ) -> torch.Tensor:
        params = sampling_params[0]
        scores = logits.float()

        temperture = params.temperature
        limits = torch.finfo(torch.float32)

        if not limits.tiny <= temperture <= limits.max:
            scores = scores.double()
            # A device divisor also avoids reciprocal overflow for tiny T.
            temperture = torch.tensor(
                temperture,
                dtype=torch.float64,
                device=logits.device,
            )

        row_max = scores.amax(dim=-1, keepdim=True)
        scores = (scores - row_max).div_(temperture)
        candidate_ids: torch.Tensor | None = None
        
        if 0 < params.top_k < scores.shape[-1]:
            scores, candidate_ids = torch.topk(
                scores,
                k=params.top_k,
                dim=-1
            )

        if scores.shape[-1] == 1:
            if candidate_ids is None:
                return torch.zeros(
                    (scores.shape[0],),
                    dtype=torch.long,
                    device=logits.device,
                )
            return candidate_ids[:, 0]
        
        # A finite maximum allows masked -inf entries, but rejects an
        # invalid distribution without scanning the vocabulary again.
        torch._assert_async(
            torch.isfinite(row_max).all(),
            "sampling logits must have a finite maximum in every row",
        )

        order: torch.Tensor | None = None

        if params.top_p < 1.0:
            scores, order = torch.sort(
                scores,
                dim=-1,
                descending=True,
                stable=True,
            )

        probabilities = torch.softmax(scores, dim=-1)
        
        if params.top_p < 1.0:
            remove = probabilities.cumsum(dim=-1) > params.top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            probabilities.masked_fill_(remove, 0.0)
            
        noise = torch.empty_like(probabilities)
        
        for row_noise, row_params, request_id in zip(
            noise,
            sampling_params,
            request_ids,
        ):
            row_noise.exponential_(
                1,
                generator=self._generator(
                    request_id,
                    row_params.seed,
                    logits.device,
                ),
            )

        choice = probabilities.div_(noise).argmax(dim=-1, keepdim=True)
        
        if order is not None:
            choice = order.gather(1, choice)
        if candidate_ids is not None:
            choice = candidate_ids.gather(1, choice)
            
        return choice.squeeze(1)


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