from __future__ import annotations

import pytest

from tinyvllm.engine import (
    Engine,
    GenerateRequest,
)
from tinyvllm.executor import CpuFakeExecutor
from tinyvllm.request import (
    RequestStatus,
    SamplingParams,
)
from tinyvllm.scheduler import Scheduler


def make_engine(
    *,
    eos_token_ids: tuple[int, ...] | None = None,
    executor=None,
) -> Engine:
    scheduler = Scheduler(
        max_active_requests=4,
        max_step_requests=4,
        max_scheduled_tokens=8,
        prefill_chunk_size=2,
        max_model_len=16,
        prefill_reserve_tokens=1,
        prefill_reserve_requests=1,
        kv_page_size=4,
        kv_num_pages=16,
        eos_token_ids=eos_token_ids,
    )

    return Engine(
        scheduler,
        CpuFakeExecutor(token_id=7) if executor is None else executor,
        clock=lambda: 0,
    )


def test_generate_returns_submission_order() -> None:
    engine = make_engine()

    outputs = engine.generate(
        (
            GenerateRequest(
                prompt_token_ids=(1, 2, 3),
                sampling=SamplingParams(
                    max_new_tokens=2,
                ),
                request_id=11,
                arrival_ns=0,
            ),
            GenerateRequest(
                prompt_token_ids=(4,),
                sampling=SamplingParams(
                    max_new_tokens=1,
                ),
                request_id=22,
                arrival_ns=0,
            ),
        )
    )

    assert [output.request_id for output in outputs] == [
        11,
        22,
    ]
    assert outputs[0].output_token_ids == (7, 7)
    assert outputs[1].output_token_ids == (7,)
    assert all(
        output.status is RequestStatus.FINISHED
        for output in outputs
    )


def test_eos_finishes_request() -> None:
    engine = make_engine(eos_token_ids=7)

    request_id = engine.submit(
        (1, 2),
        SamplingParams(max_new_tokens=8),
        arrival_ns=0,
        request_id=1,
    )

    events = engine.step(now_ns=0)

    assert events.token_events[0].request_id == request_id
    assert events.token_events[0].token_id == 7
    assert events.completion_events[0].status is RequestStatus.FINISHED
    assert events.completion_events[0].output_token_ids == (7,)


def test_ignore_eos_continues_until_length() -> None:
    engine = make_engine(eos_token_ids=7)

    request_id = engine.submit(
        (1,),
        SamplingParams(
            max_new_tokens=2,
            ignore_eos=True,
        ),
        arrival_ns=0,
        request_id=1,
    )

    first = engine.step(now_ns=0)
    second = engine.step(now_ns=1)

    assert first.completion_events == ()
    assert second.completion_events[0].request_id == request_id
    assert second.completion_events[0].output_token_ids == (7, 7)


def test_cancel_before_dispatch_emits_abort() -> None:
    engine = make_engine()

    request_id = engine.submit(
        (1, 2, 3),
        SamplingParams(max_new_tokens=4),
        arrival_ns=0,
        request_id=1,
    )

    assert engine.cancel(request_id)
    assert not engine.cancel(request_id)

    events = engine.step(now_ns=0)

    assert events.token_events == ()
    assert events.completion_events[0].request_id == request_id
    assert events.completion_events[0].status is RequestStatus.ABORTED
    assert not engine.has_unfinished()


def test_cancel_running_request_is_applied_on_next_step() -> None:
    class CancelOnExecute:
        capabilities = CpuFakeExecutor.capabilities

        def __init__(self) -> None:
            self.engine = None
            self.fake = CpuFakeExecutor(token_id=7)

        def execute(self, batch):
            assert self.engine is not None
            assert self.engine.cancel(1)
            return self.fake.execute(batch)

        def release(self, request_id: int) -> None:
            self.fake.release(request_id)

        def clear(self) -> None:
            self.fake.clear()

    executor = CancelOnExecute()
    engine = make_engine(executor=executor)
    executor.engine = engine

    request_id = engine.submit(
        (1, 2, 3),
        SamplingParams(max_new_tokens=4),
        arrival_ns=0,
        request_id=1,
    )

    first = engine.step(now_ns=0)
    assert first.token_events == ()
    assert first.completion_events == ()

    second = engine.step(now_ns=1)
    assert second.token_events == ()
    assert second.completion_events[0].request_id == request_id
    assert second.completion_events[0].status is RequestStatus.ABORTED
    assert not engine.has_unfinished()


def test_executor_failure_tears_down_dispatched_state() -> None:
    class FailingExecutor:
        capabilities = CpuFakeExecutor.capabilities

        def __init__(self) -> None:
            self.clear_calls = 0

        def execute(self, batch):
            raise ValueError("injected executor failure")

        def release(self, request_id: int) -> None:
            return None

        def clear(self) -> None:
            self.clear_calls += 1

    executor = FailingExecutor()
    engine = make_engine(executor=executor)
    engine.submit(
        (1, 2, 3),
        SamplingParams(max_new_tokens=1),
        arrival_ns=0,
        request_id=1,
    )

    with pytest.raises(RuntimeError, match="executor failed"):
        engine.step(now_ns=0)

    assert executor.clear_calls == 1
    assert not engine.scheduler.has_open_transaction
    assert engine.scheduler.kv_manager.free_page_count == (
        engine.scheduler.kv_manager.allocator.num_pages
    )

    with pytest.raises(RuntimeError, match="engine is failed"):
        engine.step(now_ns=1)


def test_zero_output_finishes_without_executor_work() -> None:
    engine = make_engine()

    output = engine.generate(
        (
            GenerateRequest(
                prompt_token_ids=(1,),
                sampling=SamplingParams(
                    max_new_tokens=0,
                ),
                request_id=9,
                arrival_ns=0,
            ),
        )
    )[0]

    assert output.status is RequestStatus.FINISHED
    assert output.output_token_ids == ()


def test_secondary_eos_finishes_request() -> None:
    engine = make_engine(
        eos_token_ids=(7, 8),
        executor=CpuFakeExecutor(token_id=8),
    )

    request_id = engine.submit(
        (1,),
        SamplingParams(max_new_tokens=8),
        arrival_ns=0,
        request_id=1,
    )

    events = engine.step(now_ns=0)

    assert events.token_events[0].request_id == request_id
    assert events.token_events[0].token_id == 8
    assert events.completion_events[0].status is RequestStatus.FINISHED


def test_non_eos_token_continues_with_multiple_eos_ids() -> None:
    engine = make_engine(
        eos_token_ids=(7, 8),
        executor=CpuFakeExecutor(token_id=9),
    )

    request_id = engine.submit(
        (1,),
        SamplingParams(max_new_tokens=2),
        arrival_ns=0,
        request_id=1,
    )

    first = engine.step(now_ns=0)
    second = engine.step(now_ns=1)

    assert first.completion_events == ()
    assert second.completion_events[0].request_id == request_id
    assert second.completion_events[0].output_token_ids == (9, 9)


def test_ignore_eos_applies_to_all_eos_ids() -> None:
    engine = make_engine(
        eos_token_ids=(7, 8),
        executor=CpuFakeExecutor(token_id=8),
    )

    request_id = engine.submit(
        (1,),
        SamplingParams(
            max_new_tokens=2,
            ignore_eos=True,
        ),
        arrival_ns=0,
        request_id=1,
    )

    first = engine.step(now_ns=0)
    second = engine.step(now_ns=1)

    assert first.completion_events == ()
    assert second.completion_events[0].request_id == request_id
    assert second.completion_events[0].output_token_ids == (8, 8)
