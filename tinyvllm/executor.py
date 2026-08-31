from __future__ import annotations

from dataclasses import dataclass

import torch

from tinyvllm._validation import (
    _check_positive_int,
    _check_token_id,
)
from tinyvllm.batch import PreparedBatch, StepResult
from tinyvllm.contiguous_kv import ContiguousKVCache
from tinyvllm.models.qwen3 import Qwen3ForCausalLM
from tinyvllm.runtime import ModelInput
from tinyvllm.sampling import TorchSampler


@dataclass(frozen=True, slots=True)
class CpuFakeExecutor:
    """Deterministic executor used by scheduler-only tests."""

    token_id: int = 0

    def __post_init__(self) -> None:
        _check_token_id(self.token_id)

    def execute(self, batch: PreparedBatch) -> StepResult:
        return StepResult(
            plan_id=batch.plan_id,
            sample_request_ids=batch.sample_request_ids,
            sampled_token_ids=(self.token_id,) * batch.num_samples,
        )


class TorchExecutor:
    """Correctness-first executor for one request and contiguous KV."""

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        *,
        max_model_len: int | None = None,
        sampler: TorchSampler | None = None,
    ) -> None:
        self.model = model.eval()
        self.sampler = (
            TorchSampler()
            if sampler is None
            else sampler
        )

        parameter = next(self.model.parameters())
        self.device = parameter.device

        config = self.model.config

        if max_model_len is None:
            max_model_len = config.max_position_embeddings

        _check_positive_int("max_model_len", max_model_len)

        if max_model_len > config.max_position_embeddings:
            raise ValueError(
                "max_model_len exceeds model position capacity"
            )

        self.cache = ContiguousKVCache(
            num_layers=config.num_hidden_layers,
            max_seq_len=max_model_len,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            device=self.device,
            dtype=parameter.dtype,
        )


    @torch.inference_mode()
    def execute(self, batch: PreparedBatch) -> StepResult:
        if batch.num_items != 1:
            raise NotImplementedError(
                "TorchExecutor currently supports one request"
            )

        model_input = ModelInput.from_prepared_batch(
            batch,
            device=self.device,
        )

        hidden_states = self.model(
            model_input.input_ids,
            model_input.positions,
            cache=self.cache,
        )

        if model_input.sample_token_indices.numel():
            logits = self.model.compute_logits(
                hidden_states,
                model_input.sample_token_indices,
            )

            sampled = self.sampler.sample(
                logits,
                batch.sampling_params,
                batch.sample_request_ids,
            )
            sampled_token_ids = tuple(
                int(token_id)
                for token_id in sampled.tolist()
            )
        else:
            sampled_token_ids = ()

        return StepResult(
            plan_id=batch.plan_id,
            sample_request_ids=batch.sample_request_ids,
            sampled_token_ids=sampled_token_ids,
        )


    def clear(self) -> None:
        self.cache.clear()
        self.sampler.clear()


class BatchedEagerExecutor:
    """Correctness-first multi-request executor."""

    def __init__(
        self,
        model: Qwen3ForCausalLM,
        *,
        max_model_len: int | None = None,
        sampler: TorchSampler | None = None,
    ) -> None:
        self.model = model.eval()
        self.sampler = (
            TorchSampler()
            if sampler is None
            else sampler
        )

        parameter = next(self.model.parameters())
        self.device = parameter.device
        config = self.model.config

        if max_model_len is None:
            max_model_len = config.max_position_embeddings

        _check_positive_int("max_model_len", max_model_len)

        if max_model_len > config.max_position_embeddings:
            raise ValueError(
                "max_model_len exceeds model position capacity"
            )

        self.max_model_len = max_model_len
        self._cache_args = {
            "num_layers": config.num_hidden_layers,
            "max_seq_len": max_model_len,
            "num_kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "device": self.device,
            "dtype": parameter.dtype,
        }
        self._caches: dict[int, ContiguousKVCache] = {}


    def _cache_for(
        self,
        request_id: int,
    ) -> ContiguousKVCache:
        cache = self._caches.get(request_id)

        if cache is None:
            cache = ContiguousKVCache(**self._cache_args)
            self._caches[request_id] = cache

        return cache


    @torch.inference_mode()
    def execute(
        self,
        batch: PreparedBatch,
    ) -> StepResult:
        request_ids = batch.request_ids

        if len(set(request_ids)) != batch.num_items:
            raise ValueError(
                "a request may appear only once in one batch or lens are not match"
            )

        model_input = ModelInput.from_prepared_batch(
            batch,
            device=self.device,
        )

        sample_hiddens: list[torch.Tensor] = []
        sample_row = 0

        for item_index, request_id in enumerate(request_ids):
            start = batch.q_offsets[item_index]
            end = batch.q_offsets[item_index + 1]

            cache = self._cache_for(request_id)

            expected_before = batch.context_lens_before[item_index]
            if cache.length != expected_before:
                raise RuntimeError(
                    "cache length does not match planned context"
                )

            hidden_states = self.model(
                model_input.input_ids[start:end],
                model_input.positions[start:end],
                cache=cache,
            )

            expected_after = batch.context_lens_after[item_index]
            if cache.length != expected_after:
                raise RuntimeError(
                    "model did not advance cache to planned context"
                )

            if (
                sample_row < batch.num_samples
                and batch.sample_token_indices[sample_row]
                == end - 1
            ):
                sample_hiddens.append(hidden_states[-1:])
                sample_row += 1

        if sample_row != batch.num_samples:
            raise RuntimeError(
                "sampling rows do not match batch item boundaries"
            )

        if sample_hiddens:
            logits = self.model.compute_logits(
                torch.cat(sample_hiddens, dim=0)
            )

            sampled = self.sampler.sample(
                logits,
                batch.sampling_params,
                batch.sample_request_ids,
            )

            sampled_token_ids = tuple(
                int(token_id)
                for token_id in sampled.tolist()
            )
        else:
            sampled_token_ids = ()

        return StepResult(
            plan_id=batch.plan_id,
            sample_request_ids=batch.sample_request_ids,
            sampled_token_ids=sampled_token_ids,
        )


    def release(self, request_id: int) -> None:
        self._caches.pop(request_id)
        self.sampler.release(request_id)


    def clear(self) -> None:
        self._caches.clear()
        self.sampler.clear()