from __future__ import annotations

from collections.abc import Iterable

from dataclasses import dataclass

import torch
from torch import nn

from tinyvllm._validation import (
    _check_positive_int,
)
from tinyvllm.batch import PreparedBatch, StepResult
from tinyvllm.runtime.runner import ModelRunner
from tinyvllm.sampling import TorchSampler


@dataclass(frozen=True, slots=True)
class ExecutorCapabilities:
    """Model geometry and execution limits required by the scheduler."""

    vocab_size: int | None = None
    max_model_len: int | None = None
    kv_block_size: int | None = None
    kv_num_blocks: int | None = None
    max_batch_size: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "max_model_len",
            "kv_block_size",
            "kv_num_blocks",
            "max_batch_size",
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
        max_batch_size: int,
    ) -> None:
        if (
            self.max_model_len is not None
            and max_model_len > self.max_model_len
        ):
            raise ValueError(
                "scheduler max_model_len exceeds executor capacity"
            )

        if (
            self.max_batch_size is not None
            and max_batch_size > self.max_batch_size
        ):
            raise ValueError(
                "scheduler maximum step batch exceeds executor capacity"
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
        max_num_seqs: int = 64,
        attention_backend: str = "cuda",
        enforce_eager: bool = False,
    ) -> None:
        self.runner = ModelRunner(
            model,
            block_size=block_size,
            num_blocks=num_blocks,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            attention_backend=attention_backend,
            enforce_eager=enforce_eager,
        )

        self.capabilities = ExecutorCapabilities(
            vocab_size=model.config.vocab_size,
            max_model_len=self.runner.max_model_len,
            kv_block_size=block_size,
            kv_num_blocks=num_blocks,
            max_batch_size=(
                self.runner.decode_inputs.capacity
                if self.runner.decode_inputs is not None
                else None
            ),
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
        try:
            self.runner.clear()
        finally:
            self.sampler.clear()
