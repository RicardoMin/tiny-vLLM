from __future__ import annotations

from dataclasses import dataclass

from tinyvllm._validation import _check_token_id
from tinyvllm.batch import PreparedBatch, StepResult


@dataclass(frozen=True, slots=True)
class CpuFakeExecutor:
    """Deterministic executor used before the model backend exists."""

    token_id: int = 0

    def __post_init__(self) -> None:
        _check_token_id(self.token_id)


    def execute(self, batch: PreparedBatch) -> StepResult:
        if not isinstance(batch, PreparedBatch):
            raise TypeError("batch must be PreparedBatch")

        return StepResult(
            plan_id=batch.plan_id,
            sample_request_ids=batch.sample_request_ids,
            sampled_token_ids=(
                self.token_id,
            ) * batch.num_samples,
        )