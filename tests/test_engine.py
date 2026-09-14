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
        kv_block_size=4,
        kv_num_blocks=16,
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
    assert engine.scheduler.block_manager.free_block_count == (
        engine.scheduler.block_manager.pool.num_blocks
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

@pytest.mark.parametrize(
    "prompt, kwargs, error",
    [
        ((), {}, ValueError),
        ((-1,), {}, ValueError),
        ((True,), {}, TypeError),
        ((1,) * 17, {}, ValueError),
        ((1,), {"arrival_ns": -1}, ValueError),
        ((1,), {"request_id": -1}, ValueError),
    ],
)
def test_invalid_submission_does_not_poison_queue_or_consume_id(prompt, kwargs, error):
    engine = make_engine()
    with pytest.raises(error):
        engine.submit(prompt, SamplingParams(max_new_tokens=1), **kwargs)
    assert not engine.has_unfinished()
    request_id = engine.submit((1,), SamplingParams(max_new_tokens=1))
    assert request_id == 0
    event, = engine.step(now_ns=0).completion_events
    assert event.request_id == 0
    assert event.submit_seq == 0
    assert event.output_token_ids == (7,)


@pytest.mark.parametrize("prompt_len", [1, 15, 16])
def test_generation_stays_within_total_model_length(prompt_len):
    engine = make_engine()
    output, = engine.generate((GenerateRequest(
        prompt_token_ids=(1,) * prompt_len,
        sampling=SamplingParams(max_new_tokens=32),
    ),))
    assert output.prompt_token_count + len(output.output_token_ids) == 16
    assert len(output.output_token_ids) == 16 - prompt_len
    assert not engine.has_unfinished()


def test_completion_cannot_be_cancelled_and_id_cannot_be_reused():
    engine = make_engine()
    request_id = engine.submit((1,), SamplingParams(max_new_tokens=1))
    engine.step(now_ns=0)
    assert not engine.cancel(request_id)
    with pytest.raises(ValueError, match="duplicate"):
        engine.submit((1,), SamplingParams(), request_id=request_id)


def test_default_eos_policy_is_immutable():
    engine = make_engine()
    assert engine.scheduler.eos_token_ids == frozenset()
    assert isinstance(engine.scheduler.eos_token_ids, frozenset)


@pytest.mark.parametrize("point", ["before_dispatch", "after_kv_dispatch", "after_kv_complete", "release_idle", "release_active", "done_clock"])
def test_all_irrecoverable_step_failures_close_ownership(point, monkeypatch):
    class Executor:
        capabilities = CpuFakeExecutor.capabilities
        def __init__(self):
            self.cleared = False
        def execute(self, batch):
            return CpuFakeExecutor(token_id=7).execute(batch)
        def release(self, request_id):
            if point.startswith("release_"):
                raise RuntimeError("injected failure")
        def clear(self):
            self.cleared = True

    from tinyvllm.cache import BlockPlanTxn, BlockTxnState
    from tinyvllm.scheduler import TxnState

    executor = Executor()
    engine = make_engine(executor=executor)
    engine.submit((1,), SamplingParams(max_new_tokens=8), request_id=1)
    if point.startswith("release_"):
        engine.step(now_ns=0)
        if point == "release_idle":
            engine.cancel(1)
        else:
            engine.submit((2,), SamplingParams(max_new_tokens=0), request_id=2)

    captured = []
    original_begin = engine.scheduler.begin_step
    def begin(now_ns):
        txn = original_begin(now_ns)
        if txn is not None:
            captured.append(txn)
        return txn
    monkeypatch.setattr(engine.scheduler, "begin_step", begin)

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    if point == "before_dispatch":
        monkeypatch.setattr(engine.scheduler, "mark_dispatched", fail)
    elif point in ("after_kv_dispatch", "after_kv_complete"):
        name = "commit_before_dispatch" if point == "after_kv_dispatch" else "complete_writes"
        original = getattr(BlockPlanTxn, name)
        def fail_after(txn):
            original(txn)
            fail()
        monkeypatch.setattr(BlockPlanTxn, name, fail_after)
    elif point == "done_clock":
        calls = 0
        def clock():
            nonlocal calls
            calls += 1
            if calls == 3:
                fail()
            return 0
        engine._clock = clock

    with pytest.raises(RuntimeError, match="injected failure"):
        engine.step(now_ns=None if point == "done_clock" else 1)
    assert executor.cleared
    assert not engine.scheduler.has_open_transaction
    manager = engine.scheduler.block_manager
    assert manager.free_block_count == manager.pool.num_blocks
    assert manager.pool.open_reservation_count == 0
    for request_id in engine.scheduler.request_ids:
        request = engine.scheduler.get_request(request_id)
        assert request.active_slot is None
        assert request.inflight_plan_id is None
        assert manager.block_table(request_id) == ()
    for txn in captured:
        if txn.state is not TxnState.COMMITTED:
            assert txn.state is TxnState.FAILED
            assert txn.kv.state is BlockTxnState.FAILED
    with pytest.raises(RuntimeError, match="engine is failed"):
        engine.step(now_ns=2)
    with pytest.raises(RuntimeError, match="scheduler is failed"):
        engine.scheduler.apply_commands((), now_ns=2)


def test_dispatch_clock_failure_rolls_back_and_can_retry():
    engine = make_engine()
    engine.submit((1,), SamplingParams(max_new_tokens=1))
    calls = 0
    def clock():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("clock failed before dispatch")
        return 0
    engine._clock = clock
    with pytest.raises(RuntimeError, match="clock failed before dispatch"):
        engine.step()
    assert not engine.scheduler.has_open_transaction
    manager = engine.scheduler.block_manager
    assert manager.free_block_count == manager.pool.num_blocks
    assert engine.step(now_ns=1).completion_events[0].output_token_ids == (7,)


def test_cleanup_error_preserves_original_failure_and_clears_executor(monkeypatch):
    class Executor:
        capabilities = CpuFakeExecutor.capabilities
        cleared = False
        def execute(self, batch):
            raise ValueError("original execution failure")
        def clear(self):
            self.cleared = True
    executor = Executor()
    engine = make_engine(executor=executor)
    engine.submit((1,), SamplingParams(max_new_tokens=1))
    def fail_cleanup(*args):
        raise RuntimeError("secondary cleanup failure")
    monkeypatch.setattr(engine.scheduler, "shutdown", fail_cleanup, raising=False)
    with pytest.raises(RuntimeError, match="executor failed") as caught:
        engine.step(now_ns=0)
    assert str(caught.value.__cause__) == "original execution failure"
    assert executor.cleared
    assert any("secondary cleanup failure" in note for note in caught.value.__cause__.__notes__)
    with pytest.raises(RuntimeError, match="engine is failed"):
        engine.submit((1,), SamplingParams())
