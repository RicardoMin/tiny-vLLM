from __future__ import annotations

from collections.abc import Iterable

from dataclasses import dataclass

import torch
from torch import nn

from tinyvllm._validation import (
    _check_positive_int,
    _check_token_id,
)
from tinyvllm.batch import PreparedBatch, StepMode, StepResult
from tinyvllm.runtime.runner import ModelRunner
from tinyvllm.sampling import TorchSampler


@dataclass(frozen=True, slots=True)
class ExecutorCapabilities:
    """Execution modes supported by one executor implementation."""

    max_batch_items: int | None = None
    supports_prefill: bool = True
    supports_decode: bool = True
    supports_mixed: bool = True

    vocab_size: int | None = None
    max_model_len: int | None = None
    kv_block_size: int | None = None
    kv_num_blocks: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_batch_items",
            "vocab_size",
            "max_model_len",
            "kv_block_size",
            "kv_num_blocks",
        ):
            value = getattr(self, name)

            if value is not None:
                _check_positive_int(name, value)


    def validate_configuration(
        self,
        *,
        max_model_len: int,
        block_size: int,
        num_blocks: int,
    ) -> None:
        if (
            self.max_model_len is not None
            and max_model_len > self.max_model_len
        ):
            raise ValueError(
                "scheduler max_model_len exceeds executor capacity"
            )

        for name, actual, expected in (
            ("block_size", block_size, self.kv_block_size),
            ("num_blocks", num_blocks, self.kv_num_blocks),
        ):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"scheduler {name} does not match executor"
                )

    def validate_token_ids(self, token_ids: Iterable[int]) -> None:
        if self.vocab_size is not None:
            if any(token_id >= self.vocab_size for token_id in token_ids):
                raise ValueError("token ID is out of vocabulary range")

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


class Executor:
    """Unified batch execution over a fixed global paged KV store."""

    def __init__(
        self,
        model: nn.Module,
        *,
        block_size: int,
        num_blocks: int,
        max_model_len: int | None = None,
        sampler: TorchSampler | None = None,
    ) -> None:
        self.runner = ModelRunner(
            model,
            block_size=block_size,
            num_blocks=num_blocks,
            max_model_len=max_model_len,
        )

        self.capabilities = ExecutorCapabilities(
            vocab_size=model.config.vocab_size,
            max_model_len=self.runner.max_model_len,
            kv_block_size=block_size,
            kv_num_blocks=num_blocks,
        )

        self.sampler = (
            TorchSampler()
            if sampler is None
            else sampler
        )


    @torch.inference_mode()
    def execute(
        self,
        batch: PreparedBatch,
    ) -> StepResult:
        logits = self.runner.run(batch)
        tokens = ()

        if logits is not None:
            sampled = self.sampler.sample(logits, batch.sampling_params, batch.sample_request_ids)
            tokens = tuple(sampled.tolist())

        return StepResult(
            batch.plan_id,
            batch.sample_request_ids,
            tokens,
        )


    def release(self, request_id: int) -> None:
        # Scheduler already returned the blocks; old tensor bytes need no wipe.
        self.sampler.release(request_id)


    def clear(self) -> None:
        self.runner.clear()
        self.sampler.clear()
