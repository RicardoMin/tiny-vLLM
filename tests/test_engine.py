from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tinyvllm.engine import (
    GenerateRequest,
)
from engine_helpers import CpuFakeExecutor
from tinyvllm.request import (
    RequestStatus,
    SamplingParams,
)
from engine_helpers import make_engine


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


@pytest.mark.parametrize("prompt_len", [1, 15])
def test_generation_stays_within_total_model_length(prompt_len):
    engine = make_engine()
    output, = engine.generate((GenerateRequest(
        prompt_token_ids=(1,) * prompt_len,
        sampling=SamplingParams(max_new_tokens=32),
    ),))
    assert output.prompt_token_count + len(output.output_token_ids) == 16
    assert len(output.output_token_ids) == 16 - prompt_len
    assert not engine.has_unfinished()


@pytest.mark.parametrize(
    "prompt_len,max_new_tokens",
    [(1, 0), (16, 4)],
)
def test_zero_output_finishes_without_executing_or_allocating_kv(
    prompt_len, max_new_tokens,
):
    executor = SimpleNamespace(
        capabilities=CpuFakeExecutor.capabilities,
        execute=Mock(side_effect=AssertionError("zero-output request executed")),
        release=Mock(),
        clear=Mock(),
    )
    engine = make_engine(executor=executor, kv_block_size=1, kv_num_blocks=1)
    try:
        output, = engine.generate((
            GenerateRequest((1,) * prompt_len, SamplingParams(
                max_new_tokens=max_new_tokens,
            )),
        ))
        assert output.output_token_ids == ()
        assert output.prompt_token_count == prompt_len
        assert output.status is RequestStatus.FINISHED
        assert engine.scheduler.block_manager.free_block_count == 1
        assert not engine.has_unfinished()
        executor.execute.assert_not_called()
        executor.release.assert_called_once_with(output.request_id)
    finally:
        engine.close()


def test_completed_request_cannot_be_cancelled_and_id_can_be_reused():
    engine = make_engine()
    request_id = engine.submit((1,), SamplingParams(max_new_tokens=1))
    engine.step(now_ns=0)
    assert not engine.cancel(request_id)
    assert engine.submit(
        (2,), SamplingParams(max_new_tokens=1), request_id=request_id,
    ) == request_id
    with pytest.raises(ValueError, match="duplicate"):
        engine.submit((3,), SamplingParams(max_new_tokens=1), request_id=request_id)


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
    assert not engine.scheduler.request_ids
    assert not engine.has_unfinished()
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

@pytest.mark.parametrize(
    "eos, token, ignore_eos, expected_count",
    [
        pytest.param(7, 7, False, 1, id="single-eos"),
        pytest.param((7, 8), 8, False, 1, id="secondary-eos"),
        pytest.param((7, 8), 9, False, 2, id="non-eos"),
        pytest.param((7, 8), 8, True, 2, id="ignore-secondary-eos"),
    ],
)
def test_eos_policy_preserves_complete_event_sequence(eos, token, ignore_eos, expected_count):
    engine = make_engine(eos_token_ids=eos, executor=CpuFakeExecutor(token_id=token))
    assert isinstance(engine.scheduler.eos_token_ids, frozenset)
    if eos is None:
        assert engine.scheduler.eos_token_ids == frozenset()
    request_id = engine.submit(
        (1, 2), SamplingParams(max_new_tokens=2, ignore_eos=ignore_eos),
        arrival_ns=0, request_id=1,
    )
    emitted = []
    for index in range(expected_count):
        events = engine.step(now_ns=index)
        token_event, = events.token_events
        assert token_event.request_id == request_id
        emitted.append(token_event.token_id)
        if index < expected_count - 1:
            assert events.completion_events == ()
    completion, = events.completion_events
    assert completion.request_id == request_id
    assert completion.status is RequestStatus.FINISHED
    assert tuple(emitted) == completion.output_token_ids == (token,) * expected_count
    assert not engine.has_unfinished()


def test_close_discards_pending_work_and_attempts_both_cleanup_paths(monkeypatch):
    engine = make_engine()
    engine.submit((1, 2), SamplingParams(max_new_tokens=2))
    engine.submit((3,), SamplingParams(max_new_tokens=1), arrival_ns=100)
    calls = []
    shutdown = engine.scheduler.shutdown

    def failed_shutdown():
        shutdown()
        calls.append('scheduler')
        raise RuntimeError('injected scheduler cleanup error')

    class Executor:
        def clear(self):
            calls.append('executor')

    monkeypatch.setattr(engine.scheduler, 'shutdown', failed_shutdown)
    engine.executor = Executor()
    with pytest.raises(RuntimeError, match='cleanup failed') as captured:
        engine.close()
    assert calls == ['scheduler', 'executor']
    assert engine.scheduler.request_ids == ()
    assert engine.scheduler.next_arrival_ns is None
    assert 'scheduler cleanup failed' in captured.value.__notes__[0]
    assert not engine.has_unfinished()
    engine.close()
    with pytest.raises(RuntimeError, match='closed'):
        engine.step()


def test_close_during_generate_is_rejected_without_discarding_the_batch():
    engine = make_engine()
    fake = CpuFakeExecutor(token_id=6)

    def execute(batch):
        with pytest.raises(RuntimeError, match='cannot close'):
            engine.close()
        return fake.execute(batch)

    engine.executor = SimpleNamespace(
        capabilities=fake.capabilities, execute=execute,
        release=fake.release, clear=fake.clear,
    )
    from tinyvllm.engine import GenerateRequest

    assert engine.generate((GenerateRequest((1,), SamplingParams(max_new_tokens=1)),))[0].output_token_ids == (6,)
    engine.close()
