from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from tinyvllm._validation import _check_non_negative_int
from tinyvllm.batch import (
    BatchBuilder,
    PreparedBatch,
    StepResult,
)
from tinyvllm.executor import ExecutorCapabilities
from tinyvllm.request import (
    CompletionEvent,
    RequestStatus,
    SamplingParams,
)
from tinyvllm.scheduler import (
    AddCommand,
    CancelCommand,
    EngineCommand,
    ScheduleDecision,
    Scheduler,
    SchedulerResourceError,
)


class EngineExecutor(Protocol):
    @property
    def capabilities(self) -> ExecutorCapabilities:
        ...


    def execute(
        self,
        batch: PreparedBatch,
    ) -> StepResult:
        ...


    def release(
        self,
        request_id: int,
    ) -> None:
        ...


    def clear(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingParams
    request_id: int | None = None
    arrival_ns: int | None = None


    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_token_ids",
            tuple(self.prompt_token_ids),
        )

        if not isinstance(self.sampling, SamplingParams):
            raise TypeError(
                "sampling must be SamplingParams"
            )


@dataclass(frozen=True, slots=True)
class TokenEvent:
    request_id: int
    token_id: int


@dataclass(frozen=True, slots=True)
class EngineStepEvents:
    token_events: tuple[TokenEvent, ...] = ()
    completion_events: tuple[CompletionEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_events",
            tuple(self.token_events),
        )
        object.__setattr__(
            self,
            "completion_events",
            tuple(self.completion_events),
        )


@dataclass(frozen=True, slots=True)
class RequestOutput:
    request_id: int
    status: RequestStatus
    output_token_ids: tuple[int, ...]
    prompt_token_count: int

    @classmethod
    def from_event(
        cls,
        event: CompletionEvent,
    ) -> "RequestOutput":
        return cls(
            request_id=event.request_id,
            status=event.status,
            output_token_ids=event.output_token_ids,
            prompt_token_count=event.prompt_token_count,
        )


class Engine:
    def __init__(
        self,
        scheduler: Scheduler,
        executor: EngineExecutor,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.scheduler = scheduler
        self.executor = executor
        self.builder = BatchBuilder(
            max_model_len=scheduler.max_model_len,
        )
        self._clock = clock

        self._commands: list[EngineCommand] = []

        self._next_seq = (
            scheduler.last_command_seq + 1
        )
        self._next_request_id = 0

        self._submitted_ids: set[int] = set()
        self._terminal_ids: set[int] = set()
        self._cancel_queued: set[int] = set()
        self._failed: BaseException | None = None


    def submit(
        self,
        prompt_token_ids: Sequence[int],
        sampling: SamplingParams,
        *,
        arrival_ns: int | None = None,
        request_id: int | None = None,
    ) -> int:
        self._ensure_healthy()

        if not isinstance(sampling, SamplingParams):
            raise TypeError(
                "sampling must be SamplingParams"
            )

        if request_id is None:
            while self._next_request_id in self._submitted_ids:
                self._next_request_id += 1

            request_id = self._next_request_id
            self._next_request_id += 1
        else:
            _check_non_negative_int(
                "request_id",
                request_id,
            )

            if request_id in self._submitted_ids:
                raise ValueError(
                    f"duplicate request_id: {request_id}"
                )

        if arrival_ns is None:
            arrival_ns = self._clock()

        _check_non_negative_int(
            "arrival_ns",
            arrival_ns,
        )

        self._commands.append(
            AddCommand(
                seq=self._next_seq,
                request_id=request_id,
                prompt_token_ids=tuple(prompt_token_ids),
                sampling=sampling,
                arrival_ns=arrival_ns,
            )
        )

        self._next_seq += 1
        self._submitted_ids.add(request_id)
        return request_id


    def cancel(self, request_id: int) -> bool:
        self._ensure_healthy()
        _check_non_negative_int(
            "request_id",
            request_id,
        )

        if request_id not in self._submitted_ids:
            return False

        if request_id in self._terminal_ids:
            return False

        if request_id in self._cancel_queued:
            return False

        has_pending_add = any(
            isinstance(command, AddCommand)
            and command.request_id == request_id
            for command in self._commands
        )

        if not has_pending_add:
            try:
                request = self.scheduler.get_request(
                    request_id
                )
            except KeyError:
                return False

            if request.terminal:
                return False

        self._commands.append(
            CancelCommand(
                seq=self._next_seq,
                request_id=request_id,
            )
        )
        self._next_seq += 1
        self._cancel_queued.add(request_id)
        return True


    def step(
        self,
        now_ns: int | None = None,
    ) -> EngineStepEvents:
        self._ensure_healthy()

        explicit_time = now_ns is not None
        start_ns = (
            self._clock()
            if now_ns is None
            else now_ns
        )
        _check_non_negative_int(
            "now_ns",
            start_ns,
        )

        cutoff = len(self._commands)
        commands = tuple(
            self._commands[:cutoff]
        )

        self.scheduler.apply_commands(
            commands,
            start_ns,
        )
        del self._commands[:cutoff]

        for command in commands:
            if isinstance(command, CancelCommand):
                self._cancel_queued.discard(
                    command.request_id
                )

        txn = self.scheduler.begin_step(start_ns)

        if txn is None:
            if self.scheduler.last_decision is ScheduleDecision.BLOCKED:
                raise SchedulerResourceError(
                    self.scheduler.blocked_reason
                )
            return self._collect_events(None)

        try:
            requests = tuple(
                self.scheduler.get_request(
                    item.request_id
                )
                for item in txn.plan.items
            )
            batch = self.builder.prepare(
                txn.plan,
                requests,
                kv=txn.kv,
            )
        except BaseException:
            self.scheduler.rollback(txn)
            raise

        try:
            self.executor.capabilities.validate(batch)
        except BaseException:
            self.scheduler.rollback(txn)
            raise

        dispatch_ns = start_ns

        if not explicit_time:
            dispatch_ns = max(
                dispatch_ns,
                self._clock(),
            )

        try:
            self.scheduler.mark_dispatched(
                txn,
                dispatch_ns,
            )
        except BaseException as exc:
            self._fail_stop(exc)
            raise

        try:
            result = self.executor.execute(batch)
        except BaseException as exc:
            self._fail_stop(exc, txn)
            raise RuntimeError(
                "executor failed; engine is failed"
            ) from exc

        done_ns = dispatch_ns

        if not explicit_time:
            done_ns = max(done_ns, self._clock())

        try:
            self.scheduler.commit_step(
                txn,
                result,
                done_ns,
            )
        except BaseException as exc:
            self._fail_stop(exc, txn)
            raise

        try:
            return self._collect_events(result)
        except BaseException as exc:
            self._fail_stop(exc)
            raise


    def has_unfinished(self) -> bool:
        return bool(
            self._commands
            or self.scheduler.request_ids
        )


    def generate(
        self,
        requests: Sequence[GenerateRequest],
    ) -> list[RequestOutput]:
        specs = tuple(requests)

        if not specs:
            return []

        request_ids = tuple(
            self.submit(
                spec.prompt_token_ids,
                spec.sampling,
                arrival_ns=spec.arrival_ns,
                request_id=spec.request_id,
            )
            for spec in specs
        )

        outputs: dict[int, RequestOutput] = {}
        target_ids = set(request_ids)

        while len(outputs) < len(request_ids):
            events = self.step()

            for event in events.completion_events:
                if event.request_id in target_ids:
                    outputs[event.request_id] = RequestOutput.from_event(event)

            if (
                not self.has_unfinished()
                and len(outputs) < len(request_ids)
            ):
                raise RuntimeError(
                    "engine stopped before all requests completed"
                )

        return [
            outputs[request_id]
            for request_id in request_ids
        ]


    def _collect_events(
        self,
        result: StepResult | None,
    ) -> EngineStepEvents:
        completion_events = (
            self.scheduler.consume_finished()
        )

        for event in completion_events:
            self.executor.release(
                event.request_id
            )
            self._terminal_ids.add(
                event.request_id
            )

        if result is None:
            return EngineStepEvents(
                completion_events=completion_events
            )

        aborted_ids = {
            event.request_id
            for event in completion_events
            if event.status is RequestStatus.ABORTED
        }

        token_events = tuple(
            TokenEvent(
                request_id=request_id,
                token_id=token_id,
            )
            for request_id, token_id in zip(
                result.sample_request_ids,
                result.sampled_token_ids,
            )
            if request_id not in aborted_ids
        )

        return EngineStepEvents(
            token_events=token_events,
            completion_events=completion_events,
        )


    def _ensure_healthy(self) -> None:
        if self._failed is not None:
            raise RuntimeError(
                "engine is failed"
            ) from self._failed


    def _fail_stop(
        self,
        cause: BaseException,
        txn=None,
    ) -> None:
        if txn is not None and self.scheduler.has_open_transaction:
            self.scheduler.fail_step(txn)

        try:
            self.executor.clear()
        except BaseException:
            pass

        self._failed = cause
