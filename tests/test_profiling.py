import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.tools import profiling, run_tiny
from benchmarks.tools.measurement import ExecutionProbe, replay
from tinyvllm.engine import Engine
from tinyvllm.executor import ExecutorCapabilities
from engine_helpers import CpuFakeExecutor
from tinyvllm.scheduler import Scheduler


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


def fake_engine(clock):
    fake = CpuFakeExecutor(token_id=7)

    def execute(batch):
        clock.now += 1_000_000
        return fake.execute(batch)

    executor = SimpleNamespace(
        capabilities=ExecutorCapabilities(vocab_size=151936),
        execute=execute,
        release=fake.release,
        clear=fake.clear,
        runner=SimpleNamespace(
            device="cuda:0",
            kv_cache=SimpleNamespace(
                key=torch.empty(256, dtype=torch.uint8),
                value=torch.empty(256, dtype=torch.uint8),
                block_size=16,
            ),
        ),
    )
    scheduler = Scheduler(
        max_active_requests=4,
        max_step_requests=4,
        max_scheduled_tokens=16,
        prefill_chunk_size=8,
        max_model_len=512,
        prefill_reserve_tokens=1,
        prefill_reserve_requests=1,
        kv_block_size=16,
        kv_num_blocks=256,
        eos_token_ids=(151645,),
    )
    return Engine(scheduler, executor, clock=clock, sleep=clock.sleep)


def record_ranges(monkeypatch, events):
    stack = []

    @contextmanager
    def tracked(name):
        stack.append(name)
        events.append(("enter", name))
        try:
            yield
        finally:
            assert stack.pop() == name
            events.append(("exit", name))

    monkeypatch.setattr(torch.cuda.nvtx, "range", tracked)
    return stack


def test_disabled_helpers_do_not_touch_cuda(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("disabled profiling touched CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    monkeypatch.setattr(torch.cuda.profiler, "start", unexpected)
    monkeypatch.setattr(torch.cuda.profiler, "stop", unexpected)
    monkeypatch.setattr(torch.cuda.nvtx, "range", unexpected)

    with profiling.capture_cuda(False, "cuda:0"):
        with profiling.nvtx_range("test", False):
            pass


@pytest.mark.parametrize("body_fails", [False, True])
def test_cleanup_always_stops_and_preserves_first_error(monkeypatch, body_fails):
    events = []
    stack = record_ranges(monkeypatch, events)
    body_error = KeyboardInterrupt("body interrupted")
    drain_error = RuntimeError("drain failed")
    stop_error = RuntimeError("stop failed")
    sync_calls = 0

    def synchronize(device):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 2:
            raise drain_error

    def stop():
        events.append("stop")
        raise stop_error

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda.profiler, "start", lambda: events.append("start"))
    monkeypatch.setattr(torch.cuda.profiler, "stop", stop)
    expected = body_error if body_fails else drain_error

    with pytest.raises(type(expected)) as raised:
        with profiling.capture_cuda(True, "cuda:0"):
            if body_fails:
                raise body_error

    assert raised.value is expected
    assert sync_calls == 2
    assert events[-1] == "stop"
    assert stack == []
    assert any("stop failed" in note for note in expected.__notes__)
    if body_fails:
        assert any("drain failed" in note for note in expected.__notes__)


def test_nvtx_step_and_mode_ranges_preserve_real_engine_results(monkeypatch):
    events = []
    stack = record_ranges(monkeypatch, events)
    records = [
        {
            "ordinal": index,
            "request_id": str(index),
            "arrival_ns": arrival,
            "prompt_token_ids": [1] * length,
            "max_new_tokens": 3,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        for index, (arrival, length) in enumerate(((0, 1), (1_000_000, 10)))
    ]
    results = []

    for enabled in (False, True):
        clock = Clock()
        engine = fake_engine(clock)
        probe = ExecutionProbe(engine.executor, engine.scheduler.block_manager.pool)
        probe.nvtx = enabled
        engine.executor = probe
        results.append(replay(
            engine, records, nvtx=enabled, clock=clock, sleep=clock.sleep,
        ))
        assert not engine.has_unfinished()
        assert probe.pool.free_block_count == probe.pool.num_blocks
        if not enabled:
            assert events == []

    assert results[0] == results[1]
    assert stack == []
    entered = [name for kind, name in events if kind == "enter"]
    assert "tinyvllm/step" in entered
    assert {name for name in entered if "/execute/" in name} == {
        "tinyvllm/execute/prefill_only",
        "tinyvllm/execute/mixed",
        "tinyvllm/execute/decode_only",
    }


@pytest.mark.parametrize("profile", [False, True])
def test_cli_captures_only_first_measurement_and_labels_report(
    tmp_path, monkeypatch, profile,
):
    clock = Clock()
    engine = fake_engine(clock)
    output = tmp_path / "report.json"
    root = Path(__file__).resolve().parents[1]
    argv = [
        "run_tiny", "--model", str(tmp_path),
        "--workload", str(root / "benchmarks/workloads/smoke-synth-qwen3-v1.json"),
        "--output", str(output), "--warmup", "2", "--repeats", "3",
    ]
    monkeypatch.setattr(sys, "argv", argv + (["--profile"] if profile else []))
    monkeypatch.setattr(run_tiny.Engine, "from_config", lambda config: engine)
    monkeypatch.setattr(run_tiny, "git_identity", lambda path: {})
    monkeypatch.setattr(run_tiny, "model_manifest", lambda path: {})
    monkeypatch.setattr(run_tiny, "packages", lambda: [])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "mock")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=4096))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda device: 1024)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    ranges = []
    stack = record_ranges(monkeypatch, ranges)
    active = False
    lifecycle = []

    def start():
        nonlocal active
        assert not active
        active = True
        lifecycle.append("start")

    def stop():
        nonlocal active
        assert active
        active = False
        lifecycle.append("stop")

    monkeypatch.setattr(torch.cuda.profiler, "start", start)
    monkeypatch.setattr(torch.cuda.profiler, "stop", stop)
    observed = []

    def observed_replay(engine, records, **kwargs):
        observed.append((active, kwargs.get("nvtx", False), engine.executor.nvtx))
        return replay(engine, records, clock=clock, sleep=clock.sleep, **kwargs)

    monkeypatch.setattr(run_tiny, "replay", observed_replay)
    run_tiny.main()

    expected = [False, False, profile, False, False]
    assert observed == [(flag, flag, flag) for flag in expected]
    assert lifecycle == (["start", "stop"] if profile else [])
    assert stack == []
    assert not active
    assert not engine.executor.nvtx
    report = json.loads(output.read_text())
    assert report["status"] == "ok"
    assert report["purpose"] == ("diagnostic" if profile else "benchmark")
    assert report["protocol"]["cuda_capture_repeat"] == (0 if profile else None)
    assert [run["cuda_capture_requested"] for run in report["runs"]] == [profile, False, False]
    assert all(run["output_tokens"] == 64 for run in report["runs"])
