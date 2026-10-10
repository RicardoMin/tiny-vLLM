from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from tinyvllm._validation import (
    _check_non_negative_int,
)
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


if TYPE_CHECKING:
    from tinyvllm.config import EngineConfig

class Engine:
    """A serial engine; generate() owns its request batch exclusively."""

    def __init__(
        self,
        scheduler: Scheduler,
        executor: EngineExecutor,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        executor.capabilities.validate_configuration(
            max_model_len=scheduler.max_model_len,
            block_size=scheduler.kv_block_size,
            num_blocks=scheduler.block_manager.pool.num_blocks,
        )
        executor.capabilities.validate_token_ids(
            scheduler.eos_token_ids
        )

        self.scheduler = scheduler
        self.executor = executor
        self.builder = BatchBuilder(
            max_model_len=scheduler.max_model_len,
        )

        self._clock = clock
        self._sleep = sleep
        self._commands: list[EngineCommand] = []
        self._next_seq = (
            scheduler.last_command_seq + 1
        )
        self._active_ids = set(scheduler.request_ids)
        self._next_request_id = max(self._active_ids, default=-1) + 1
        self._cancel_queued: set[int] = set()
        self._closed = False
        self._failed: str | None = None
        self._generating = False


    @classmethod
    def from_config(cls, config: EngineConfig) -> Engine:
        from tinyvllm.bootstrap import create_engine

        return create_engine(config)


    def submit(
        self,
        prompt_token_ids: Sequence[int],
        sampling: SamplingParams,
        *,
        arrival_ns: int | None = None,
        request_id: int | None = None,
    ) -> int:
        self._ensure_healthy()

        if self._generating:
            raise RuntimeError(
                "generate() owns the engine until it returns"
            )

        specs = (
            GenerateRequest(
                prompt_token_ids=tuple(prompt_token_ids),
                sampling=sampling,
                request_id=request_id,
                arrival_ns=arrival_ns,
            ),
        )

        commands, next_id = self._prepare_commands(specs)
        self._publish_commands(commands, next_id)

        return commands[0].request_id


    def _prepare_commands(
        self,
        specs: tuple[GenerateRequest, ...],
    ) -> tuple[tuple[AddCommand, ...], int]:
        commands: list[AddCommand] = []
        staged_ids: set[int] = set()
        next_id = self._next_request_id

        for spec in specs:
            if not isinstance(spec, GenerateRequest):
                raise TypeError(
                    "requests must contain GenerateRequest values"
                )

            self.scheduler.validate_request(
                spec.prompt_token_ids,
                spec.sampling,
            )
            self.executor.capabilities.validate_token_ids(
                spec.prompt_token_ids,
            )

            arrival_ns = (
                self._clock()
                if spec.arrival_ns is None
                else spec.arrival_ns
            )
            _check_non_negative_int("arrival_ns", arrival_ns)

            request_id = (
                next_id
                if spec.request_id is None
                else spec.request_id
            )
            _check_non_negative_int("request_id", request_id)

            if (
                request_id in self._active_ids
                or request_id in staged_ids
            ):
                raise ValueError(
                    f"duplicate active request_id: {request_id}"
                )

            commands.append(
                AddCommand(
                    seq=self._next_seq + len(commands),
                    request_id=request_id,
                    prompt_token_ids=spec.prompt_token_ids,
                    sampling=spec.sampling,
                    arrival_ns=arrival_ns,
                )
            )

            staged_ids.add(request_id)
            next_id = max(next_id, request_id + 1)

        return tuple(commands), next_id


    def _publish_commands(
        self,
        commands: tuple[AddCommand, ...],
        next_id: int,
    ) -> None:
        self._commands.extend(commands)
        self._active_ids.update(
            command.request_id
            for command in commands
        )
        self._next_seq += len(commands)
        self._next_request_id = next_id


    def cancel(self, request_id: int) -> bool:
        self._ensure_healthy()
        _check_non_negative_int(
            "request_id",
            request_id,
        )

        if (
            request_id not in self._active_ids
            or request_id in self._cancel_queued
        ):
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
            try:
                return self._collect_events(None)
            except BaseException as exc:
                self._fail_stop(exc)
                raise

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

            dispatch_ns = start_ns
            if not explicit_time:
                dispatch_ns = max(dispatch_ns, self._clock())
        except BaseException:
            self.scheduler.rollback(txn)
            raise

        try:
            self.scheduler.mark_dispatched(txn, dispatch_ns)
        except BaseException as exc:
            self._fail_stop(exc)
            raise

        try:
            result = self.executor.execute(batch)
        except BaseException as exc:
            self._fail_stop(exc)
            raise RuntimeError("executor failed; engine is failed") from exc

        try:
            done_ns = dispatch_ns
            if not explicit_time:
                done_ns = max(done_ns, self._clock())
            self.scheduler.commit_step(txn, result, done_ns)
            return self._collect_events(result)
        except BaseException as exc:
            self._fail_stop(exc)
            raise

    def has_unfinished(self) -> bool:
        return bool(self._active_ids)


    def generate(
        self,
        requests: Sequence[GenerateRequest],
    ) -> list[RequestOutput]:
        self._ensure_healthy()

        if self._generating or self.has_unfinished():
            raise RuntimeError(
                "generate() requires an idle engine"
            )

        commands, next_id = self._prepare_commands(tuple(requests))
        if not commands: 
            return []

        request_ids = tuple(
            command.request_id
            for command in commands
        )
        self._publish_commands(commands, next_id)

        outputs: dict[int, RequestOutput] = {}
        self._generating = True

        try:
            while self.has_unfinished():
                events = self.step()

                for event in events.completion_events:
                    outputs[event.request_id] = (
                        RequestOutput.from_event(event)
                    )

                if self.scheduler.last_decision is ScheduleDecision.IDLE:
                    arrival_ns = self.scheduler.next_arrival_ns

                    if arrival_ns is not None:
                        delay = (
                            arrival_ns - self._clock()
                        ) / 1_000_000_000

                        if delay > 0:
                            self._sleep(delay)
        finally:
            self._generating = False

        if len(outputs) != len(request_ids):
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
        completions = (
            self.scheduler.consume_finished()
        )
        completed_ids = {
            event.request_id
            for event in completions
        }

        # A cancellation queued during execute may target a request that just
        # finished. Retire it before that request ID becomes reusable.
        if completed_ids & self._cancel_queued:
            remaining = [
                command 
                for command in self._commands
                if not (
                    isinstance(command, CancelCommand)
                    and command.request_id in completed_ids
                )
            ]

            first_seq = self.scheduler.last_command_seq + 1
            self._commands = [
                replace(command, seq=first_seq + index)
                for index, command in enumerate(remaining)
            ]
            self._next_seq = first_seq + len(remaining)

        self._active_ids.difference_update(completed_ids)
        self._cancel_queued.difference_update(completed_ids)

        for event in completions:
            self.executor.release(event.request_id)

        if result is None:
            return EngineStepEvents(
                completion_events=completions
            )

        tokens = tuple(
            TokenEvent(
                request_id=request_id,
                token_id=token_id,
            )
            for request_id, token_id in zip(
                result.sample_request_ids,
                result.sampled_token_ids,
            )
        )

        return EngineStepEvents(
            token_events=tokens,
            completion_events=completions,
        )

        
    def reset_prefix_cache(self) -> None:
        """Drop reusable prefixes between workloads, without reallocating KV."""
        self._ensure_healthy()
        
        if self._generating or self.has_unfinished():
            raise RuntimeError(
                "reset_prefix_cache requires an idle engine"
            )

        self.scheduler.block_manager.reset_prefix_cache()


    def close(self) -> None:
        """Permanently dispose resources; pending requests are discarded."""
        if self._closed:
            return
        if self._generating:
            raise RuntimeError("cannot close while generate() is running")

        self._closed = True
        error = RuntimeError("engine cleanup failed")
        self._dispose(error)
        if getattr(error, "__notes__", None):
            raise error


    def _ensure_healthy(self) -> None:
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._failed is not None:
            raise RuntimeError(
                f"engine is failed: {self._failed}"
            )

    def _fail_stop(
        self,
        cause: BaseException,
    ) -> None:
        # Keep text only: retaining cause would retain its traceback and KV.
        self._failed = f"{type(cause).__name__} : {cause}"
        self._dispose(cause)


    def _dispose(self, cause: BaseException) -> None:
        self._commands.clear()
        self._active_ids.clear()
        self._cancel_queued.clear()

        for name, cleanup in (
            ("scheduler", self.scheduler.shutdown),
            ("executor", self.executor.clear),
        ):
            try:
                cleanup()
            except BaseException as cleanup_error:
                cause.add_note(f"{name} cleanup failed: {cleanup_error!r}")
