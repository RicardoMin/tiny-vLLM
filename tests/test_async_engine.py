from __future__ import annotations

import asyncio
from pathlib import Path
import threading

import pytest

from tinyvllm.async_engine import (
    AsyncEngine,
    OverloadedError,
    SlowConsumerError,
    WorkerError,
)
from tinyvllm.config import EngineConfig
from tinyvllm.engine import Engine, TokenEvent
from tinyvllm.executor import CpuFakeExecutor, ExecutorCapabilities
from tinyvllm.request import CompletionEvent, RequestStatus, SamplingParams
from tinyvllm.scheduler import Scheduler


class Gate:
    """A bounded rendezvous, never a timing-based sleep."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.resume = threading.Event()

    def block(self) -> None:
        self.entered.set()
        if not self.resume.wait(5):
            raise RuntimeError("test gate was not released")


async def reached(event: threading.Event) -> None:
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, 5), 6)


async def checkpoint() -> None:
    ready = asyncio.Event()
    asyncio.get_running_loop().call_soon(ready.set)
    await ready.wait()


async def collect(stream):
    async with stream:
        return [event async for event in stream]


def run(coroutine):
    async def bounded():
        return await asyncio.wait_for(coroutine, 15)

    return asyncio.run(bounded())


class Harness:
    def __init__(self, monkeypatch, tmp_path: Path) -> None:
        self.config = EngineConfig(
            model=tmp_path,
            max_model_len=16,
            max_num_seqs=4,
            max_num_batched_tokens=8,
            prefill_chunk_size=2,
            prefill_reserve_tokens=1,
            prefill_reserve_requests=1,
        )
        self.calls = []
        self.batches = []
        self.factory_gate = None
        self.clear_gate = None
        self.execute_gates = {}
        self.fail_factory = False
        self.fail_execute = False
        self.engine = None
        harness = self

        class Executor:
            capabilities = ExecutorCapabilities(vocab_size=32)

            def execute(self, batch):
                harness.record("execute")
                harness.batches.append(batch)
                gate = harness.execute_gates.get(len(harness.batches))
                if gate is not None:
                    gate.block()
                if harness.fail_execute:
                    raise RuntimeError("injected execution failure")
                return CpuFakeExecutor(token_id=7).execute(batch)

            def release(self, request_id):
                harness.record("release", request_id)

            def clear(self):
                harness.record("clear")
                if harness.clear_gate is not None:
                    harness.clear_gate.block()

        class TracedEngine(Engine):
            def submit(self, prompt_token_ids, sampling, **kwargs):
                harness.record("submit")
                if prompt_token_ids[0] == 31:
                    raise TypeError("injected request validation failure")
                return super().submit(prompt_token_ids, sampling, **kwargs)

            def step(self, **kwargs):
                harness.record("step")
                return super().step(**kwargs)

            def cancel(self, request_id):
                harness.record("cancel", request_id)
                return super().cancel(request_id)

            def close(self):
                harness.record("close")
                return super().close()

        def factory(cls, config):
            self.record("factory")
            if self.factory_gate is not None:
                self.factory_gate.block()
            if self.fail_factory:
                raise RuntimeError("injected load failure")
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
            )
            self.engine = TracedEngine(scheduler, Executor())
            return self.engine

        monkeypatch.setattr(Engine, "from_config", classmethod(factory))

    def record(self, name, request_id=None):
        self.calls.append((name, threading.get_ident(), request_id))

    def assert_owner(self, loop_thread):
        owners = {thread_id for _, thread_id, _ in self.calls}
        assert len(owners) == 1
        assert loop_thread not in owners
        assert sum(name == "factory" for name, _, _ in self.calls) == 1
        assert sum(name == "close" for name, _, _ in self.calls) == 1


def test_owner_thread_dynamic_batch_and_order(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)
    gate = harness.execute_gates[1] = Gate()

    async def scenario():
        bridge = AsyncEngine(harness.config)
        try:
            assert await asyncio.gather(bridge.start(), bridge.start()) == [bridge, bridge]
            first = bridge.submit((1,), SamplingParams(max_new_tokens=3))
            await reached(gate.entered)
            second = bridge.submit((2,), SamplingParams(max_new_tokens=1))
            gate.resume.set()
            outputs = await asyncio.gather(collect(first), collect(second))
            for events, count in zip(outputs, (3, 1)):
                assert all(isinstance(event, TokenEvent) for event in events[:-1])
                assert [event.token_id for event in events[:-1]] == [7] * count
                terminal = events[-1]
                assert isinstance(terminal, CompletionEvent)
                assert terminal.output_token_ids == (7,) * count
                assert terminal.status is RequestStatus.FINISHED
            assert any(
                set(batch.request_ids) == {first.request_id, second.request_id}
                and 0 in batch.context_lens_before
                and any(length > 0 for length in batch.context_lens_before)
                for batch in harness.batches
            )
        finally:
            gate.resume.set()
            await bridge.aclose()
        assert not bridge._slots
        assert bridge._reserved_tokens == 0
        harness.assert_owner(threading.get_ident())

    run(scenario())


@pytest.mark.parametrize("limits", [{"max_requests": 1}, {"max_reserved_tokens": 2}])
def test_completed_unread_result_keeps_admission_credit(monkeypatch, tmp_path, limits):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with AsyncEngine(harness.config, **limits) as bridge:
            stream = bridge.submit((1,), SamplingParams(max_new_tokens=1))
            await asyncio.wait_for(stream._slot.signal.wait(), 5)
            assert stream._slot.retired
            with pytest.raises(OverloadedError):
                bridge.submit((2,), SamplingParams(max_new_tokens=1))
            assert isinstance(await anext(stream), TokenEvent)
            with pytest.raises(OverloadedError):
                bridge.submit((2,), SamplingParams(max_new_tokens=1))
            assert isinstance(await anext(stream), CompletionEvent)
            assert bridge._reserved_tokens == 0
            assert len(await collect(bridge.submit((2,), SamplingParams(max_new_tokens=1)))) == 2

    run(scenario())


@pytest.mark.parametrize("max_new_tokens", [0, 1])
def test_terminal_has_its_own_slot_even_when_token_buffer_is_full(
    monkeypatch, tmp_path, max_new_tokens,
):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with AsyncEngine(harness.config, max_buffered_tokens=1) as bridge:
            stream = bridge.submit((1,), SamplingParams(max_new_tokens=max_new_tokens))
            await asyncio.wait_for(stream._slot.signal.wait(), 5)
            assert stream._slot.retired
            assert len(stream._slot.tokens) == max_new_tokens
            assert isinstance(stream._slot.terminal, CompletionEvent)
            events = await collect(stream)
            assert len(events) == max_new_tokens + 1
            assert events[-1].output_token_ids == (7,) * max_new_tokens
            assert events[-1].status is RequestStatus.FINISHED
            assert not bridge._slots
            assert bridge._reserved_tokens == 0

    run(scenario())


def test_context_capped_credit_and_oversized_input_rejection(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with AsyncEngine(harness.config, max_reserved_tokens=16) as bridge:
            with pytest.raises(ValueError):
                bridge.submit((1,) * 17)
            with pytest.raises(TypeError):
                bridge.submit(iter((1, 2)))
            with pytest.raises(ValueError):
                bridge.submit((True,))
            with pytest.raises(TypeError):
                bridge.submit((1,), object())
            assert bridge._reserved_tokens == 0
            stream = bridge.submit((1,) * 15, SamplingParams(max_new_tokens=64))
            assert stream._slot.cost == 16
            assert isinstance((await collect(stream))[-1], CompletionEvent)

    run(scenario())


@pytest.mark.parametrize("prompt,error", [((999,), ValueError), ((31,), TypeError)])
def test_owner_validation_error_isolated_to_request(monkeypatch, tmp_path, prompt, error):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with AsyncEngine(harness.config) as bridge:
            invalid = bridge.submit(prompt, SamplingParams(max_new_tokens=1))
            valid = bridge.submit((1,), SamplingParams(max_new_tokens=1))
            with pytest.raises(error):
                await collect(invalid)
            assert isinstance((await collect(valid))[-1], CompletionEvent)
            assert not bridge._slots
            assert bridge._reserved_tokens == 0

    run(scenario())


def test_slow_consumer_cancels_only_its_request(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)
    gate = harness.execute_gates[1] = Gate()

    async def scenario():
        bridge = await AsyncEngine(harness.config, max_buffered_tokens=1).start()
        try:
            slow = bridge.submit((1,), SamplingParams(max_new_tokens=5))
            await reached(gate.entered)
            healthy = bridge.submit((2,), SamplingParams(max_new_tokens=1))
            gate.resume.set()
            events = await collect(healthy)
            assert events[-1].output_token_ids == (7,)
            with pytest.raises(SlowConsumerError, match="overflow"):
                await collect(slow)
            assert not slow._slot.tokens
            subsequent = bridge.submit((3,), SamplingParams(max_new_tokens=1))
            assert isinstance((await collect(subsequent))[-1], CompletionEvent)
            assert any(
                name == "cancel" and rid == slow.request_id
                for name, _, rid in harness.calls
            )
        finally:
            gate.resume.set()
            await bridge.aclose()

    run(scenario())


@pytest.mark.parametrize("cancel_reader", [False, True])
def test_abandoned_stream_wakes_reader_and_waits_for_owner_retirement(
    monkeypatch, tmp_path, cancel_reader,
):
    harness = Harness(monkeypatch, tmp_path)
    gate = harness.execute_gates[1] = Gate()

    async def scenario():
        bridge = await AsyncEngine(harness.config, max_requests=1).start()
        retired = threading.Event()
        original_retire = bridge._retire

        def record_retirement(slot):
            original_retire(slot)
            if slot.retired and slot.released:
                retired.set()

        monkeypatch.setattr(bridge, "_retire", record_retirement)
        try:
            stream = bridge.submit((1,), SamplingParams(max_new_tokens=5))
            await reached(gate.entered)
            reader = asyncio.create_task(anext(stream))
            await checkpoint()
            with pytest.raises(RuntimeError, match="one reader"):
                await anext(stream)
            if cancel_reader:
                reader.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await reader
            else:
                async with stream:
                    pass
                with pytest.raises(StopAsyncIteration):
                    await asyncio.wait_for(reader, 5)
            with pytest.raises(OverloadedError):
                bridge.submit((2,), SamplingParams(max_new_tokens=1))
            gate.resume.set()
            await reached(retired)
            assert stream.request_id not in bridge._slots
            assert stream.request_id not in bridge._dirty
            assert bridge._reserved_tokens == 0
            assert any(name == "cancel" and rid == stream.request_id for name, _, rid in harness.calls)
            assert len(await collect(bridge.submit((2,), SamplingParams(max_new_tokens=1)))) == 2
        finally:
            gate.resume.set()
            await bridge.aclose()

    run(scenario())


def test_notifications_are_coalesced_and_retired_ids_are_removed(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with AsyncEngine(harness.config, max_buffered_tokens=8) as bridge:
            loop = asyncio.get_running_loop()
            original_schedule = loop.call_soon_threadsafe
            original_retire = bridge._retire
            posted = []
            retired_ids = set()
            all_complete = threading.Event()

            def schedule(callback, *args, **kwargs):
                if callback == bridge._wake_readers:
                    posted.append(callback)
                return original_schedule(callback, *args, **kwargs)

            def retire(slot):
                original_retire(slot)
                if slot.retired:
                    retired_ids.add(slot.request_id)
                    if len(retired_ids) == 2:
                        all_complete.set()

            monkeypatch.setattr(loop, "call_soon_threadsafe", schedule)
            monkeypatch.setattr(bridge, "_retire", retire)
            streams = [bridge.submit((1,), SamplingParams(max_new_tokens=8)) for _ in range(2)]
            # Deliberately hold the loop until the worker publishes both full results.
            # This gate proves coalescing without depending on scheduler timing.
            assert all_complete.wait(5)
            assert len(posted) == 1
            assert len(bridge._dirty) == 2
            for stream in streams:
                assert len(stream._slot.tokens) == 8
                assert len(await collect(stream)) == 9
            assert not bridge._slots
            assert not bridge._dirty
            await checkpoint()
            assert not bridge._notification_pending

    run(scenario())


def test_startup_cancellation_closes_shared_owner(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)
    gate = harness.factory_gate = Gate()

    async def scenario():
        bridge = AsyncEngine(harness.config)
        first = asyncio.create_task(bridge.start())
        second = asyncio.create_task(bridge.start())
        try:
            await reached(gate.entered)
            with pytest.raises(RuntimeError, match="not running"):
                bridge.submit((1,))
            first.cancel()
            await checkpoint()
            assert bridge._closing
        finally:
            gate.resume.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(RuntimeError, match="closed"):
            await second
        await bridge.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await bridge.start()
        assert not bridge._thread.is_alive()
        harness.assert_owner(threading.get_ident())

    run(scenario())


def test_startup_failure_is_reported_and_joined(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)
    harness.fail_factory = True

    async def scenario():
        bridge = AsyncEngine(harness.config)
        with pytest.raises(WorkerError, match="injected load failure"):
            await bridge.start()
        with pytest.raises(WorkerError, match="injected load failure"):
            await bridge.aclose()
        assert not bridge._thread.is_alive()

    run(scenario())


def test_thread_start_failure_still_allows_close(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)

    def fail_start(thread):
        raise RuntimeError("injected thread start failure")

    async def scenario():
        bridge = AsyncEngine(harness.config)
        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "start", fail_start)
            with pytest.raises(RuntimeError, match="injected thread start failure"):
                await bridge.start()
        assert bridge._thread is None
        assert not harness.calls
        await asyncio.wait_for(bridge.aclose(), 1)
        with pytest.raises(RuntimeError, match="closed"):
            await bridge.start()

    run(scenario())


def test_execution_failure_reaches_active_and_queued_requests(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)
    harness.fail_execute = True
    gate = harness.execute_gates[1] = Gate()

    async def scenario():
        bridge = await AsyncEngine(harness.config).start()
        try:
            active = bridge.submit((1,), SamplingParams(max_new_tokens=5))
            await reached(gate.entered)
            queued = bridge.submit((2,), SamplingParams(max_new_tokens=1))
            gate.resume.set()
            for stream in (active, queued):
                with pytest.raises(WorkerError, match="injected execution failure"):
                    await collect(stream)
        finally:
            gate.resume.set()
            with pytest.raises(WorkerError, match="injected execution failure"):
                await bridge.aclose()
        assert not bridge._slots
        assert not bridge._thread.is_alive()
        harness.assert_owner(threading.get_ident())

    run(scenario())


def test_cancelling_close_wait_does_not_cancel_cleanup(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)
    execute = harness.execute_gates[1] = Gate()
    clear = harness.clear_gate = Gate()

    async def scenario():
        bridge = await AsyncEngine(harness.config).start()
        try:
            stream = bridge.submit((1,), SamplingParams(max_new_tokens=5))
            await reached(execute.entered)
            reader = asyncio.create_task(anext(stream))
            await checkpoint()
            closing = asyncio.create_task(bridge.aclose())
            with pytest.raises(RuntimeError, match="closed"):
                await asyncio.wait_for(reader, 5)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            execute.resume.set()
            await reached(clear.entered)
            assert bridge._thread.is_alive()
            assert not bridge._close_task.done()
        finally:
            execute.resume.set()
            clear.resume.set()
            await bridge.aclose()
        assert not bridge._thread.is_alive()
        assert bridge._reserved_tokens == 0
        harness.assert_owner(threading.get_ident())

    run(scenario())


def test_foreign_event_loop_is_rejected_without_damaging_owner(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with AsyncEngine(harness.config) as bridge:
            stream = bridge.submit((1,), SamplingParams(max_new_tokens=1))

            async def foreign():
                with pytest.raises(RuntimeError, match="event loop"):
                    await bridge.start()
                with pytest.raises(RuntimeError, match="event loop"):
                    bridge.submit((2,))
                with pytest.raises(RuntimeError, match="event loop"):
                    await anext(stream)
                with pytest.raises(RuntimeError, match="event loop"):
                    await bridge.aclose()

            await asyncio.to_thread(lambda: asyncio.run(foreign()))
            assert isinstance((await collect(stream))[-1], CompletionEvent)

    run(scenario())
