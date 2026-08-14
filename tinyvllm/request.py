from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto


def _check_non_negative_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _check_token_id(token_id: int) -> None:
    _check_non_negative_int("token_id", token_id)


class RequestStatus(Enum):
    PENDING = auto()
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    ABORTED = auto()

    @property
    def terminal(self) -> bool:
        return self in (RequestStatus.FINISHED, RequestStatus.ABORTED)


class WorkClass(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 0.0
    ignore_eos: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        _check_non_negative_int("max_new_tokens", self.max_new_tokens)
        _check_non_negative_int("seed", self.seed)

        if type(self.temperature) not in (int, float):
            raise TypeError("temperature must be a number")
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        object.__setattr__(self, "temperature", temperature)

        if type(self.ignore_eos) is not bool:
            raise TypeError("ignore_eos must be bool")


    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0



@dataclass(slots=True)
class RequestState:
    request_id: int
    submit_seq: int
    prompt_token_ids: list[int]
    sampling: SamplingParams
    arrival_ns: int

    output_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.PENDING
    work_class: WorkClass = WorkClass.PREFILL
    computed_tokens: int = 0
    active_slot: int | None = None
    inflight_plan_id: int | None = None
    cancel_requested: bool = False

    admitted_ns: int | None = None
    first_token_ns: int | None = None
    finished_ns: int | None = None

    preemption_count: int = 0
    recomputed_tokens: int = 0

    def __post_init__(self) -> None:
        self.prompt_token_ids = list(self.prompt_token_ids)
        self.output_token_ids = list(self.output_token_ids)
        self.check_committed_invariants()


    @property
    def context_token_ids(self) -> list[int]:
        # Debug/test convenience only.
        # BatchBuilder will use token_slice() on the hot path.
        return self.prompt_token_ids + self.output_token_ids


    @property
    def total_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)


    @property
    def terminal(self) -> bool:
        return self.status.terminal


    def token_slice(self, start: int, end: int) -> list[int]:
        _check_non_negative_int("start", start)
        _check_non_negative_int("end", end)

        if start > end or end > self.total_tokens:
            raise ValueError("invalid token slice")

        prompt_len = len(self.prompt_token_ids)

        if end <= prompt_len:
            return self.prompt_token_ids[start:end]

        if start >= prompt_len:
            return self.output_token_ids[
                start - prompt_len : end - prompt_len
            ]

        return (
            self.prompt_token_ids[start:]
            + self.output_token_ids[: end - prompt_len]
        )


    @property
    def num_uncomputed_tokens(self) -> int:
        return self.total_tokens - self.computed_tokens


    @property
    def remaining_new_tokens(self) -> int:
        return self.sampling.max_new_tokens - self.num_output_tokens


    def check_committed_invariants(self) -> None:
        _check_non_negative_int("request_id", self.request_id)
        _check_non_negative_int("submit_seq", self.submit_seq)
        _check_non_negative_int("arrival_ns", self.arrival_ns)
        _check_non_negative_int("computed_tokens", self.computed_tokens)
        _check_non_negative_int("preemption_count", self.preemption_count)
        _check_non_negative_int("recomputed_tokens", self.recomputed_tokens)

        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")

        for token_id in self.prompt_token_ids:
            _check_token_id(token_id)
        for token_id in self.output_token_ids:
            _check_token_id(token_id)

        if not isinstance(self.sampling, SamplingParams):
            raise TypeError("sampling must be SamplingParams")
        if not isinstance(self.status, RequestStatus):
            raise TypeError("status must be RequestStatus")
        if not isinstance(self.work_class, WorkClass):
            raise TypeError("work_class must be WorkClass")
        if type(self.cancel_requested) is not bool:
            raise TypeError("cancel_requested must be bool")

        if self.computed_tokens > self.total_tokens:
            raise ValueError("computed_tokens exceeds known context")

        if self.num_output_tokens > self.sampling.max_new_tokens:
            raise ValueError("output exceeds max_new_tokens")

        self._check_optional_index("active_slot", self.active_slot)
        self._check_optional_index(
            "inflight_plan_id", self.inflight_plan_id
        )

        for name in ("admitted_ns", "first_token_ns", "finished_ns"):
            self._check_optional_timestamp(name, getattr(self, name))

        if self.status is RequestStatus.RUNNING:
            if self.active_slot is None:
                raise ValueError("running request must own an active slot")
            if self.admitted_ns is None:
                raise ValueError("running request must have admitted_ns")
        elif self.active_slot is not None or self.inflight_plan_id is not None:
            raise ValueError("only running requests may own runtime resources")

        if self.status is RequestStatus.PENDING:
            if self.admitted_ns is not None:
                raise ValueError("pending request cannot be admitted")
            if self.output_token_ids or self.computed_tokens != 0:
                raise ValueError("pending request cannot have progress")
            if self.work_class is not WorkClass.PREFILL:
                raise ValueError("pending request must be prefill work")

        if self.status is RequestStatus.WAITING:
            if self.computed_tokens != 0:
                raise ValueError("waiting request cannot own valid KV progress")
            if self.work_class is not WorkClass.PREFILL:
                raise ValueError("waiting request must be prefill work")

        if self.terminal:
            if self.finished_ns is None:
                raise ValueError("terminal request must have finished_ns")
            if (
                self.status is RequestStatus.FINISHED
                and self.admitted_ns is None
            ):
                raise ValueError("finished request must have admitted_ns")
        elif self.finished_ns is not None:
            raise ValueError("non-terminal request cannot have finished_ns")

        if self.output_token_ids and self.first_token_ns is None:
            raise ValueError("output tokens require first_token_ns")
        if not self.output_token_ids and self.first_token_ns is not None:
            raise ValueError("first_token_ns requires an output token")

        if self.admitted_ns is not None:
            if self.admitted_ns < self.arrival_ns:
                raise ValueError("admitted_ns cannot precede arrival_ns")

        if self.first_token_ns is not None:
            if self.admitted_ns is None:
                raise ValueError("first token requires admission")
            if self.first_token_ns < self.admitted_ns:
                raise ValueError("first_token_ns cannot precede admission")

        if self.finished_ns is not None:
            progress_times = [
                timestamp
                for timestamp in (self.admitted_ns, self.first_token_ns)
                if timestamp is not None
            ]

            # A future request may be aborted before its arrival timestamp.
            if self.status is RequestStatus.FINISHED:
                progress_times.append(self.arrival_ns)

            if progress_times and self.finished_ns < max(progress_times):
                raise ValueError("finished_ns is earlier than request progress")

        if not self.terminal:
            if (
                self.status is RequestStatus.RUNNING
                and self.remaining_new_tokens == 0
            ):
                raise ValueError("running request has no output budget")

            if self.computed_tokens == self.total_tokens:
                raise ValueError("non-terminal request must have work remaining")

            if self.work_class is WorkClass.DECODE:
                if not self.output_token_ids:
                    raise ValueError("decode work requires a sampled output token")
                if self.computed_tokens + 1 != self.total_tokens:
                    raise ValueError(
                        "decode request must have exactly one pending token"
                    )


    @staticmethod
    def _check_optional_index(
        name: str,
        value: int | None,
    ) -> None:
        if value is not None:
            _check_non_negative_int(name, value)


    @staticmethod
    def _check_optional_timestamp(
        name: str,
        value: int | None,
    ) -> None:
        if value is not None:
            _check_non_negative_int(name, value)