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