"""Small synchronous Engine fixture shared by API and admission tests."""

from tinyvllm.engine import Engine
from dataclasses import dataclass

from tinyvllm.batch import PreparedBatch, StepResult
from tinyvllm.executor import ExecutorCapabilities
from tinyvllm.scheduler import Scheduler


@dataclass(frozen=True, slots=True)
class CpuFakeExecutor:
    """Deterministic executor used by scheduler-only tests."""

    token_id: int = 0

    capabilities = ExecutorCapabilities()

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


def make_engine(
    executor=None,
    *,
    eos_token_ids=None,
    kv_block_size=4,
    kv_num_blocks=16,
    **engine_options,
):
    scheduler = Scheduler(
        max_active_requests=4,
        max_step_requests=4,
        max_scheduled_tokens=8,
        prefill_chunk_size=2,
        max_model_len=16,
        prefill_reserve_tokens=1,
        prefill_reserve_requests=1,
        kv_block_size=kv_block_size,
        kv_num_blocks=kv_num_blocks,
        eos_token_ids=eos_token_ids,
    )
    engine_options.setdefault("clock", lambda: 0)
    return Engine(
        scheduler,
        CpuFakeExecutor(token_id=7) if executor is None else executor,
        **engine_options,
    )
