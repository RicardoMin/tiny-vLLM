import gc
import weakref

import pytest

from tinyvllm.engine import Engine, GenerateRequest
from tinyvllm.executor import CpuFakeExecutor, Executor
from tinyvllm.request import RequestStatus, SamplingParams
from tinyvllm.scheduler import Scheduler


def make_engine(executor=None, **kwargs):
    scheduler = Scheduler(
        max_active_requests=4,
        max_step_requests=4,
        max_scheduled_tokens=8,
        prefill_chunk_size=2,
        max_model_len=16,
        prefill_reserve_tokens=1,
        prefill_reserve_requests=1,
        kv_block_size=2,
        kv_num_blocks=32,
    )
    return Engine(
        scheduler,
        executor or CpuFakeExecutor(token_id=7),
        clock=kwargs.pop('clock', lambda: 0),
        **kwargs,
    )


def test_completed_ids_do_not_accumulate_and_explicit_ids_can_be_reused():
    engine = make_engine()
    for request_id in range(100):
        output, = engine.generate((
            GenerateRequest((1,), SamplingParams(max_new_tokens=1)),
        ))
        assert output.request_id == request_id
        assert not engine._active_ids
        assert not engine.scheduler.request_ids

    engine.submit((2,), SamplingParams(max_new_tokens=1), request_id=0)
    event, = engine.step(now_ns=0).completion_events
    assert event.request_id == 0
    assert engine.submit((3,), SamplingParams(max_new_tokens=1)) == 100


def test_generate_does_not_consume_an_existing_stream():
    engine = make_engine()
    request_id = engine.submit((1,), SamplingParams(max_new_tokens=1))
    with pytest.raises(RuntimeError, match='idle engine'):
        engine.generate((GenerateRequest((2,), SamplingParams()),))

    event, = engine.step(now_ns=0).completion_events
    assert event.request_id == request_id
    assert event.output_token_ids == (7,)


@pytest.mark.parametrize('bad', ['prompt', 'duplicate', 'type', 'arrival'])
def test_generate_prepares_all_requests_before_publishing_any(bad):
    first = GenerateRequest((1,), SamplingParams(max_new_tokens=1))
    second = {
        'prompt': GenerateRequest((-1,), SamplingParams()),
        'duplicate': GenerateRequest((2,), SamplingParams(), request_id=0),
        'type': object(),
        'arrival': GenerateRequest((2,), SamplingParams(), arrival_ns=-1),
    }[bad]
    engine = make_engine()
    with pytest.raises((TypeError, ValueError)):
        engine.generate((first, second))

    assert not engine.has_unfinished()
    assert engine.submit((3,), SamplingParams(max_new_tokens=1)) == 0
    event, = engine.step(now_ns=0).completion_events
    assert event.submit_seq == 0


def test_generate_sleeps_until_future_arrival_instead_of_busy_polling():
    now_ns = 0
    waits = []

    def sleep(seconds):
        nonlocal now_ns
        waits.append(seconds)
        now_ns += round(seconds * 1_000_000_000)

    engine = make_engine(clock=lambda: now_ns, sleep=sleep)
    output, = engine.generate((
        GenerateRequest((1,), SamplingParams(max_new_tokens=1), arrival_ns=2_000_000_000),
    ))
    assert waits == [2.0]
    assert output.output_token_ids == (7,)


def test_completion_retires_stale_cancel_and_resequences_mixed_commands():
    class EnqueueDuringExecute(CpuFakeExecutor):
        engine = None
        queued = False

        def execute(self, batch):
            if not self.queued:
                object.__setattr__(self, 'queued', True)
                assert self.engine.cancel(7)
                self.engine.submit((2,), SamplingParams(max_new_tokens=1), request_id=8)
                assert self.engine.cancel(8)
                self.engine.submit((3,), SamplingParams(max_new_tokens=1), request_id=9)
            return super().execute(batch)

    executor = EnqueueDuringExecute(token_id=7)
    engine = make_engine(executor)
    object.__setattr__(executor, 'engine', engine)
    engine.submit((1,), SamplingParams(max_new_tokens=1), request_id=7)
    first, = engine.step(now_ns=0).completion_events
    assert first.request_id == 7
    assert first.status is RequestStatus.FINISHED

    engine.submit((4,), SamplingParams(max_new_tokens=1), request_id=7)
    events = {event.request_id: event for event in engine.step(now_ns=1).completion_events}
    assert events[8].status is RequestStatus.ABORTED
    assert events[9].output_token_ids == (7,)
    assert events[7].output_token_ids == (7,)
    assert not engine.has_unfinished()


def test_generate_rejects_foreign_submission_during_execution():
    class SubmitDuringExecute(CpuFakeExecutor):
        engine = None

        def execute(self, batch):
            with pytest.raises(RuntimeError, match='owns the engine'):
                self.engine.submit((2,), SamplingParams())
            return super().execute(batch)

    executor = SubmitDuringExecute()
    engine = make_engine(executor)
    object.__setattr__(executor, 'engine', engine)
    assert len(engine.generate((GenerateRequest((1,), SamplingParams(max_new_tokens=1)),))) == 1
    assert not engine.has_unfinished()


def eager_engine():
    from paged_helpers import models

    model, _ = models()
    executor = Executor(model, block_size=2, num_blocks=32, max_model_len=16)
    return model, make_engine(executor)


def test_out_of_vocabulary_submission_does_not_fail_an_existing_request():
    model, engine = eager_engine()
    request_id = engine.submit((1,), SamplingParams(max_new_tokens=2))
    assert engine.step(now_ns=0).token_events
    with pytest.raises(ValueError, match='vocabulary'):
        engine.submit((model.config.vocab_size,), SamplingParams())

    completion, = engine.step(now_ns=1).completion_events
    assert completion.request_id == request_id
    assert len(completion.output_token_ids) == 2
    assert engine.executor.runner.kv_cache is not None


@pytest.mark.parametrize('mismatch', ['length', 'block_size', 'num_blocks'])
def test_low_level_engine_rejects_incompatible_executor_at_construction(mismatch):
    from paged_helpers import models

    model, _ = models()
    executor = Executor(
        model,
        block_size=4 if mismatch == 'block_size' else 2,
        num_blocks=16 if mismatch == 'num_blocks' else 32,
        max_model_len=8 if mismatch == 'length' else 16,
    )
    with pytest.raises(ValueError, match='scheduler'):
        make_engine(executor)


def test_fail_stop_releases_actual_kv_objects_after_exception_scope_ends(monkeypatch):
    model, engine = eager_engine()
    store_ref = weakref.ref(engine.executor.runner.kv_cache)
    key_ref = weakref.ref(engine.executor.runner.kv_cache.key)

    def fail(*args, **kwargs):
        raise RuntimeError('injected layer failure')

    monkeypatch.setattr(model.model.layers[-1], 'forward', fail)
    engine.submit((1,), SamplingParams(max_new_tokens=1))

    def execute_and_catch():
        with pytest.raises(RuntimeError, match='executor failed'):
            engine.step(now_ns=0)

    execute_and_catch()
    gc.collect()
    assert store_ref() is None
    assert key_ref() is None
    assert not engine.has_unfinished()
    assert not engine.scheduler.request_ids
    with pytest.raises(RuntimeError, match='engine is failed'):
        engine.step(now_ns=1)
