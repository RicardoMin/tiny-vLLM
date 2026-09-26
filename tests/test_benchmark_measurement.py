from dataclasses import replace

import pytest

from benchmarks.tools.measurement import ExecutionProbe, replay
from tinyvllm.engine import Engine, EngineStepEvents, TokenEvent
from engine_helpers import CpuFakeExecutor
from tinyvllm.request import RequestStatus, SamplingParams
from tinyvllm.scheduler import Scheduler


MILLISECOND = 1_000_000


class FakeClock:
    def __init__(self):
        self.now_ns = 7_000_000_000
        self.waits = []

    def __call__(self):
        return self.now_ns

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now_ns += round(seconds * 1_000_000_000)


class TimedExecutor:
    capabilities = CpuFakeExecutor.capabilities

    def __init__(self, clock, durations_ms):
        self.clock = clock
        self.durations_ms = iter(durations_ms)
        self.fake = CpuFakeExecutor(token_id=7)
        self.released = []
        self.clear_calls = 0

    def execute(self, batch):
        self.clock.now_ns += next(self.durations_ms) * MILLISECOND
        return self.fake.execute(batch)

    def release(self, request_id):
        self.released.append(request_id)

    def clear(self):
        self.clear_calls += 1


def make_engine(clock, durations_ms, *, eos_token_ids=()):
    scheduler = Scheduler(
        max_active_requests=4,
        max_step_requests=4,
        max_scheduled_tokens=16,
        prefill_chunk_size=8,
        max_model_len=16,
        prefill_reserve_tokens=1,
        prefill_reserve_requests=1,
        kv_block_size=2,
        kv_num_blocks=32,
        eos_token_ids=eos_token_ids,
    )
    executor = TimedExecutor(clock, durations_ms)
    probe = ExecutionProbe(executor, scheduler.block_manager.pool)
    engine = Engine(scheduler, probe, clock=clock, sleep=clock.sleep)
    return engine, probe, executor


def record(ordinal, *, arrival_ms=0, prompt=(1,), output=1):
    return {
        "ordinal": ordinal,
        "request_id": f"workload-{ordinal}",
        "arrival_ns": arrival_ms * MILLISECOND,
        "prompt_token_ids": list(prompt),
        "max_new_tokens": output,
        "temperature": 0.0,
        "ignore_eos": True,
    }


def test_replay_measures_fixed_arrivals_token_gaps_and_peak_before_release():
    clock = FakeClock()
    engine, probe, executor = make_engine(clock, [10, 30, 20, 5])
    records = [
        record(0, prompt=(1, 2, 3), output=3),
        record(1, arrival_ms=15),
        record(2, arrival_ms=100),
    ]

    result = replay(engine, records, clock=clock, sleep=clock.sleep)

    first, second, third = result["requests"]
    assert [row["request_id"] for row in result["requests"]] == [
        "workload-0", "workload-1", "workload-2",
    ]
    assert first["token_times_ns"] == [10_000_000, 40_000_000, 60_000_000]
    assert first["token_ids"] == [7, 7, 7]
    assert first["ttft_ms"] == 10
    assert first["tpot_ms"] == 25
    assert first["e2e_ms"] == 60

    # Arrival at 15 ms stays fixed although a synchronous step ends at 40 ms.
    assert second["arrival_ns"] == 15_000_000
    assert second["submitted_ns"] == 40_000_000
    assert second["token_times_ns"] == [60_000_000]
    assert second["ttft_ms"] == second["e2e_ms"] == 45
    assert second["tpot_ms"] is None
    assert third["submitted_ns"] == 100_000_000
    assert third["token_times_ns"] == [105_000_000]
    assert third["ttft_ms"] == 5
    assert third["tpot_ms"] is None
    assert clock.waits == [0.04]

    assert result["duration_ns"] == 105_000_000
    assert result["completed_requests"] == 3
    assert result["completion_rate"] == 1.0
    assert result["output_tokens"] == 5
    assert result["output_tokens_per_second"] == pytest.approx(5 / 0.105)
    assert result["ttft_ms"] == {"count": 3, "p50": 10, "p95": 45}
    assert result["itl_ms"] == {"count": 2, "p50": 20, "p95": 30}
    assert result["tpot_ms"] == {"count": 1, "p50": 25, "p95": 25}
    assert result["e2e_ms"] == {"count": 3, "p50": 45, "p95": 60}
    assert result["submit_lag_ms"] == {"count": 3, "p50": 0, "p95": 25}

    # Step 3 holds 3 blocks for request 0 and 1 for request 1; both then finish.
    assert probe.peak_used_blocks == 4
    assert probe.pool.free_block_count == probe.pool.num_blocks == 32
    assert probe.steps_by_mode == {
        "PREFILL_ONLY": 2,
        "DECODE_ONLY": 1,
        "MIXED": 1,
    }
    assert sorted(executor.released) == [0, 1, 2]
    probe.reset()
    assert probe.peak_used_blocks == 0
    assert not probe.steps_by_mode
    probe.clear()
    assert executor.clear_calls == 1


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("token_after_completion", "unknown or completed"),
        ("unknown_request", "unknown or completed"),
        ("duplicate_token", "duplicate token event"),
        ("duplicate_completion", "unknown or completed"),
        ("wrong_sequence", "complete output sequence"),
        ("missing_terminal_token", "complete output sequence"),
        ("aborted", "did not finish successfully"),
        ("missing_completion", "missing completion"),
    ],
)
def test_replay_rejects_invalid_event_streams(monkeypatch, corruption, message):
    clock = FakeClock()
    engine, _, _ = make_engine(clock, [10, 10])
    original_step = engine.step
    steps = 0

    def corrupt_step():
        nonlocal steps
        steps += 1
        events = original_step()
        tokens = events.token_events
        completions = events.completion_events

        if corruption == "token_after_completion" and steps == 2:
            tokens += (TokenEvent(request_id=0, token_id=9),)
        elif corruption == "unknown_request":
            tokens += (TokenEvent(request_id=99, token_id=9),)
        elif corruption == "duplicate_token":
            tokens += (tokens[0],)
        elif corruption == "duplicate_completion":
            completions += (completions[0],)
        elif corruption == "wrong_sequence":
            completions = (replace(completions[0], output_token_ids=(9,)),)
        elif corruption == "missing_terminal_token":
            tokens = tuple(token for token in tokens if token.request_id != 0)
        elif corruption == "aborted":
            completions = (replace(completions[0], status=RequestStatus.ABORTED),)
        elif corruption == "missing_completion":
            completions = ()

        return EngineStepEvents(tokens, completions)

    monkeypatch.setattr(engine, "step", corrupt_step)
    records = [record(0), record(1, output=2)]
    with pytest.raises(RuntimeError, match=message):
        replay(engine, records, clock=clock, sleep=clock.sleep)


def test_replay_prevalidates_all_requests_without_submitting_any():
    clock = FakeClock()
    engine, probe, _ = make_engine(clock, [])
    records = [record(0), record(1, prompt=(1,) * 15, output=2)]

    with pytest.raises(ValueError, match="truncated by max_model_len"):
        replay(engine, records, clock=clock, sleep=clock.sleep)

    assert not engine.has_unfinished()
    assert not probe.steps_by_mode
    assert engine.submit((1,), SamplingParams(max_new_tokens=1)) == 0


def test_single_token_run_has_no_itl_or_tpot_distribution():
    clock = FakeClock()
    engine, _, _ = make_engine(clock, [5])

    result = replay(engine, [record(0)], clock=clock, sleep=clock.sleep)

    assert result["requests"][0]["token_ids"] == [7]
    assert result["requests"][0]["finished_ns"] == 5_000_000
    assert result["itl_ms"] == {"count": 0, "p50": None, "p95": None}
    assert result["tpot_ms"] == {"count": 0, "p50": None, "p95": None}


@pytest.mark.parametrize("ignore_eos, expected_count", [(False, 1), (True, 3)])
def test_replay_obeys_eos_policy(ignore_eos, expected_count):
    clock = FakeClock()
    engine, _, _ = make_engine(clock, [5, 5, 5], eos_token_ids=(7,))
    request = record(0, output=3)
    request["ignore_eos"] = ignore_eos

    result = replay(engine, [request], clock=clock, sleep=clock.sleep)

    assert result["requests"][0]["token_ids"] == [7] * expected_count
    assert result["output_tokens"] == expected_count
    assert result["completion_rate"] == 1.0


@pytest.mark.parametrize("corruption", ["early_without_eos", "token_after_eos"])
def test_replay_rejects_invalid_eos_sequences(monkeypatch, corruption):
    clock = FakeClock()
    eos_id = 7 if corruption == "early_without_eos" else 42
    engine, _, _ = make_engine(clock, [5, 5], eos_token_ids=(eos_id,))
    original_step = engine.step
    request = record(0, output=2)
    request["ignore_eos"] = False
    steps = 0

    def corrupt_step():
        nonlocal steps
        steps += 1
        events = original_step()
        if corruption == "early_without_eos":
            return EngineStepEvents(
                (TokenEvent(0, 9),),
                (replace(events.completion_events[0], output_token_ids=(9,)),),
            )
        if steps == 1:
            return EngineStepEvents((TokenEvent(0, 42),), ())
        return EngineStepEvents(
            events.token_events,
            (replace(events.completion_events[0], output_token_ids=(42, 7)),),
        )

    monkeypatch.setattr(engine, "step", corrupt_step)
    message = "early completion without EOS" if corruption == "early_without_eos" else "tokens emitted after EOS"
    with pytest.raises(RuntimeError, match=message):
        replay(engine, [request], clock=clock, sleep=clock.sleep)


@pytest.mark.parametrize("token_id", [-1, True, 16])
def test_replay_rejects_invalid_output_token_ids(monkeypatch, token_id):
    clock = FakeClock()
    engine, probe, _ = make_engine(clock, [5])
    probe.capabilities = replace(probe.capabilities, vocab_size=16)
    original_step = engine.step

    def corrupt_step():
        events = original_step()
        return EngineStepEvents((TokenEvent(0, token_id),), events.completion_events)

    monkeypatch.setattr(engine, "step", corrupt_step)
    with pytest.raises(RuntimeError, match="token ID"):
        replay(engine, [record(0)], clock=clock, sleep=clock.sleep)
