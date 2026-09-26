from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
import threading
import time
import traceback


from tinyvllm.config import EngineConfig
from tinyvllm.engine import Engine, TokenEvent
from tinyvllm.request import CompletionEvent, SamplingParams


class OverloadedError(RuntimeError):
    pass


class SlowConsumerError(RuntimeError):
    pass


class WorkerError(RuntimeError):
    pass


@dataclass(slots=True)
class _Slot:
    request_id: int
    prompt: tuple[int, ...]
    sampling: SamplingParams
    cost: int
    arrival_ns: int
    signal: asyncio.Event = field(default_factory=asyncio.Event)
    tokens: deque[TokenEvent] = field(default_factory=deque)
    terminal: CompletionEvent | tuple[type[Exception], str] | None = None
    retired: bool = False
    released: bool = False
    reading: bool = False


class RequestStream:
    """One reader; use async with to cancel on early exit."""

    def __init__(self, owner: AsyncEngine, slot: _Slot) -> None:
        self._owner = owner
        self._slot = slot


    @property
    def request_id(self) -> int:
        return self._slot.request_id


    def __aiter__(self) -> RequestStream:
        return self


    async def __anext__(self) -> TokenEvent | CompletionEvent:
        owner, slot = self._owner, self._slot
        owner._check_loop()
        if slot.reading:
            raise RuntimeError("a request stream supports only one reader")
        slot.reading = True
        try:
            while True:
                with owner._condition:
                    if slot.released:
                        raise StopAsyncIteration
                    if slot.tokens:
                        return slot.tokens.popleft()
                    if slot.terminal is not None:
                        terminal = slot.terminal
                        slot.terminal = None
                        slot.released = True
                        owner._retire(slot)
                        if isinstance(terminal, tuple):
                            error_type, message = terminal
                            raise error_type(message)
                        return terminal
                    slot.signal.clear()
                await slot.signal.wait()
        except asyncio.CancelledError:
            await self.aclose()
            raise
        finally:
            slot.reading = False


    async def aclose(self) -> None:
        owner, slot = self._owner, self._slot
        owner._check_loop()
        with owner._condition:
            if not slot.released:
                slot.released = True
                slot.tokens.clear()
                slot.terminal = None
                if not slot.retired:
                    owner._cancels.add(slot.request_id)
                owner._retire(slot)
                slot.signal.set()
                owner._condition.notify()


    async def __aenter__(self) -> RequestStream:
        self._owner._check_loop()
        return self


    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()


class AsyncEngine:
    """Bounded token bridge. Only the worker ever touches the GPU engine."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        max_requests: int = 64,
        max_reserved_tokens: int = 65536,
        max_buffered_tokens: int = 128,
    ) -> None:
        for value in (max_requests, max_reserved_tokens, max_buffered_tokens):
            if type(value) is not int or value <= 0:
                raise ValueError("bridge limits must be positive integers")
        self._config = config
        self._max_requests = max_requests
        self._max_reserved_tokens = max_reserved_tokens
        self._max_buffered_tokens = max_buffered_tokens
        self._condition = threading.Condition()
        self._slots: dict[int, _Slot] = {}
        self._pending: deque[_Slot] = deque()
        self._cancels: set[int] = set()
        self._dirty: set[int] = set()
        self._notification_pending = False
        self._reserved_tokens = 0
        self._next_id = 0
        self._closing = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready: Future[str | None] = Future()
        self._done: Future[str | None] = Future()
        self._close_task: asyncio.Task | None = None


    def _check_loop(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("use the event loop that started this bridge")


    async def start(self) -> AsyncEngine:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        self._check_loop()
        with self._condition:
            if self._closing:
                raise RuntimeError("async engine is closed")
            if self._thread is None:
                thread = threading.Thread(
                    target=self._run, name="tinyvllm-owner", daemon=False,
                )
                thread.start()
                self._thread = thread
        try:
            error = await asyncio.shield(asyncio.wrap_future(self._ready))
            if error is not None:
                raise WorkerError(error)
            with self._condition:
                if self._closing:
                    raise RuntimeError("async engine closed during startup")
            return self
        except BaseException as cause:
            try:
                await self.aclose()
            except BaseException as cleanup_error:
                cause.add_note(f"startup cleanup: {cleanup_error!r}")
            raise


    def submit(
        self,
        prompt_token_ids: Sequence[int],
        sampling: SamplingParams | None = None,
    ) -> RequestStream:
        self._check_loop()
        sampling = SamplingParams() if sampling is None else sampling
        if not isinstance(sampling, SamplingParams):
            raise TypeError("sampling must be SamplingParams")
        if not isinstance(prompt_token_ids, Sequence):
            raise TypeError("prompt_token_ids must be a sequence")
        length = len(prompt_token_ids)
        if not 0 < length <= self._config.max_model_len:
            raise ValueError("prompt length exceeds the supported context")

        cost = min(length + sampling.max_new_tokens, self._config.max_model_len)
        with self._condition:
            if self._closing or not self._ready.done():
                raise RuntimeError("async engine is not running")
            if (
                len(self._slots) >= self._max_requests
                or self._reserved_tokens + cost > self._max_reserved_tokens
            ):
                raise OverloadedError("request or token admission budget exhausted")

            prompt = tuple(prompt_token_ids)
            if len(prompt) != length or any(
                type(token) is not int or token < 0 for token in prompt
            ):
                raise ValueError("prompt must contain nonnegative integer token IDs")

            slot = _Slot(self._next_id, prompt, sampling, cost, time.monotonic_ns())
            self._next_id += 1
            self._slots[slot.request_id] = slot
            self._pending.append(slot)
            self._reserved_tokens += cost
            self._condition.notify()
        return RequestStream(self, slot)


    def _retire(self, slot: _Slot) -> None:
        # Both sides must relinquish ownership before admission credit returns.
        if slot.retired and slot.released:
            if self._slots.pop(slot.request_id, None) is slot:
                self._reserved_tokens -= slot.cost

            self._dirty.discard(slot.request_id)
            self._cancels.discard(slot.request_id)


    def _notify(self, slot: _Slot) -> None:
        if slot.released:
            return
        self._dirty.add(slot.request_id)

        if not self._notification_pending:
            self._notification_pending = True
            try:
                self._loop.call_soon_threadsafe(self._wake_readers)
            except RuntimeError:
                if not self._loop.is_closed():
                    raise
                self._notification_pending = False
                self._closing = True
                self._condition.notify()


    def _wake_readers(self) -> None:
        with self._condition:
            self._notification_pending = False
            for request_id in self._dirty:
                slot = self._slots.get(request_id)

                if slot is not None:
                    slot.signal.set()

            self._dirty.clear()


    def _fail(self, slot: _Slot, error_type: type[Exception], message: str) -> None:
        if not slot.released and slot.terminal is None:
            slot.tokens.clear()
            slot.terminal = (error_type, message)
            self._notify(slot)


    def _run(self) -> None:
        engine = None
        failure = None

        try:
            engine = Engine.from_config(self._config)
            self._ready.set_result(None)
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: (
                            self._closing
                            or self._pending
                            or self._cancels
                            or engine.has_unfinished()
                        )
                    )
                    if self._closing:
                        break
                    pending = tuple(self._pending)
                    self._pending.clear()
                    cancels = tuple(self._cancels)
                    self._cancels.clear()

                for slot in pending:
                    with self._condition:
                        if slot.released or self._closing:
                            slot.prompt = ()
                            slot.retired = True
                            self._retire(slot)
                            continue
                    try:
                        engine.submit(
                            slot.prompt, slot.sampling,
                            request_id=slot.request_id, arrival_ns=slot.arrival_ns,
                        )
                    except (TypeError, ValueError) as error:
                        with self._condition:
                            error_type = TypeError if isinstance(error, TypeError) else ValueError
                            self._fail(slot, error_type, str(error))
                            slot.retired = True
                            self._retire(slot)
                    finally:
                        slot.prompt = ()

                for request_id in cancels:
                    engine.cancel(request_id)

                with self._condition:
                    if self._closing:
                        continue

                if not engine.has_unfinished():
                    continue
                events = engine.step()
                with self._condition:
                    # A step can carry the final token and its completion together.
                    for event in events.token_events:
                        slot = self._slots[event.request_id]
                        if slot.released or slot.terminal is not None:
                            continue
                        if len(slot.tokens) == self._max_buffered_tokens:
                            self._fail(slot, SlowConsumerError, "token output buffer overflow")
                            self._cancels.add(slot.request_id)
                        else:
                            slot.tokens.append(event)
                            self._notify(slot)

                    for event in events.completion_events:
                        slot = self._slots[event.request_id]
                        slot.retired = True
                        if not slot.released and slot.terminal is None:
                            slot.terminal = event
                            self._notify(slot)
                        self._retire(slot)
        except BaseException as error:
            failure = "".join(traceback.format_exception(error)).strip()
        finally:
            with self._condition:
                self._closing = True
                self._pending.clear()
                self._cancels.clear()
                for slot in tuple(self._slots.values()):
                    slot.prompt = ()
                    slot.retired = True
                    self._fail(
                        slot,
                        WorkerError if failure else RuntimeError,
                        failure or "async engine closed",
                    )
                    self._retire(slot)
            try:
                if engine is not None:
                    engine.close()
            except BaseException as error:
                cleanup = "".join(traceback.format_exception(error)).strip()
                failure = f"{failure}; cleanup: {cleanup}" if failure else cleanup
            if not self._ready.done():
                self._ready.set_result(failure)
            self._done.set_result(failure)


    async def _wait_closed(self) -> None:
        error = await asyncio.shield(asyncio.wrap_future(self._done))
        await asyncio.to_thread(self._thread.join)

        with self._condition:
            for slot in self._slots.values():
                slot.signal.set()
            self._slots.clear()
            self._dirty.clear()
            self._reserved_tokens = 0
        if error is not None:
            raise WorkerError(error)


    async def aclose(self) -> None:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        self._check_loop()

        with self._condition:
            self._closing = True
            for slot in self._slots.values():
                self._fail(slot, RuntimeError, "async engine closed")

            self._condition.notify()

        if self._thread is not None:
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._wait_closed())
                self._close_task.add_done_callback(
                    lambda task: task.exception() if not task.cancelled() else None
                )

            await asyncio.shield(self._close_task)


    async def __aenter__(self) -> AsyncEngine:
        return await self.start()


    async def __aexit__(self, exc_type, exc, traceback) -> None:
        try:
            await self.aclose()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"async engine cleanup: {cleanup_error!r}")