from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tinyvllm._validation import (
    _check_positive_int,
    _check_token_id,
)
from tinyvllm.batch import PreparedBatch, StepMode, StepResult
from tinyvllm.contiguous_kv import ContiguousKVCache
from tinyvllm.runtime.runner import ModelRunner
from tinyvllm.sampling import TorchSampler


@dataclass(frozen=True, slots=True)
class ExecutorCapabilities:
    """Execution modes supported by one executor implementation."""

    max_batch_items: int | None = None
    supports_prefill: bool = True
    supports_decode: bool = True
    supports_mixed: bool = True

    def __post_init__(self) -> None:
        if self.max_batch_items is not None:
            _check_positive_int(
                "max_batch_items",
                self.max_batch_items,
            )

    def validate(self, batch: PreparedBatch) -> None:
        if (
            self.max_batch_items is not None
            and batch.num_items > self.max_batch_items
        ):
            raise RuntimeError(
                "executor cannot handle this batch size"
            )

        supported = {
            StepMode.PREFILL_ONLY: self.supports_prefill,
            StepMode.DECODE_ONLY: self.supports_decode,
            StepMode.MIXED: self.supports_mixed,
        }[batch.mode]
        if not supported:
            raise RuntimeError(
                f"executor does not support {batch.mode.name.lower()}"
            )


@dataclass(frozen=True, slots=True)
class CpuFakeExecutor:
    """Deterministic executor used by scheduler-only tests."""

    token_id: int = 0

    capabilities = ExecutorCapabilities()

    def __post_init__(self) -> None:
        _check_token_id(self.token_id)

    def execute(self, batch: PreparedBatch) -> StepResult:
        return StepResult(
            plan_id=batch.plan_id,
            sample_request_ids=batch.sample_request_ids,
            sampled_token_ids=(self.token_id,) * batch.num_samples,
        )

    def release(self, request_id: int) -> None:
        return None

    def clear(self) -> None:
        return None


class EagerExecutor:
    """Correctness-first executor for one or more requests."""

    capabilities = ExecutorCapabilities()

    def __init__(
        self,
        model: nn.Module,
        *,
        max_model_len: int | None = None,
        sampler: TorchSampler | None = None,
    ) -> None:
        self.runner = ModelRunner(
            model,
            max_model_len=max_model_len,
        )
        self.sampler = (
            TorchSampler()
            if sampler is None
            else sampler
        )

    @property
    def model(self) -> nn.Module:
        return self.runner.model

    @property
    def device(self) -> torch.device:
        return self.runner.device

    @property
    def max_model_len(self) -> int:
        return self.runner.max_model_len

    @property
    def _caches(self) -> dict[int, ContiguousKVCache]:
        return self.runner._caches


    @torch.inference_mode()
    def execute(
        self,
        batch: PreparedBatch,
    ) -> StepResult:
        logits = self.runner.run(batch)
        if logits is not None:
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
        self.runner.release(request_id)
        self.sampler.release(request_id)


    def clear(self) -> None:
        self.runner.clear()
        self.sampler.clear()
