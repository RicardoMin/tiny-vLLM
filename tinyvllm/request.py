from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

from tinyvllm._validation import (
    _check_non_negative_int,
    _check_token_id,
)


_MAX_TORCH_SEED = (1 << 64) - 1

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
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        _check_non_negative_int("max_new_tokens", self.max_new_tokens)
        _check_non_negative_int("seed", self.seed)
        _check_non_negative_int("top_k", self.top_k)

        if self.seed > _MAX_TORCH_SEED:
            raise ValueError("seed must fit in 64 bits")

        if type(self.temperature) not in (int, float):
            raise TypeError("temperature must be a number")
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        object.__setattr__(self, "temperature", temperature)

        if type(self.top_p) not in (int, float):
            raise TypeError("top_p must be a number")
        top_p = float(self.top_p)
        if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        object.__setattr__(self, "top_p", top_p)

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

    def __post_init__(self) -> None:
        self.prompt_token_ids = list(self.prompt_token_ids)
        self.output_token_ids = list(self.output_token_ids)

        _check_non_negative_int("request_id", self.request_id)
        _check_non_negative_int("submit_seq", self.submit_seq)
        _check_non_negative_int("arrival_ns", self.arrival_ns)

        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if not isinstance(self.sampling, SamplingParams):
            raise TypeError("sampling must be SamplingParams")

        for token_id in self.prompt_token_ids:
            _check_token_id(token_id)
        for token_id in self.output_token_ids:
            _check_token_id(token_id)


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


@dataclass(frozen=True, slots=True)
class CompletionEvent:
    """Immutable terminal result delivered by the scheduler."""

    request_id: int
    status: RequestStatus
    output_token_ids: tuple[int, ...]
    prompt_token_count: int
    submit_seq: int
    arrival_ns: int
    admitted_ns: int | None
    first_token_ns: int | None
    finished_ns: int
    computed_tokens: int

    def __post_init__(self) -> None:
        _check_non_negative_int("request_id", self.request_id)
        _check_non_negative_int("prompt_token_count", self.prompt_token_count)
        _check_non_negative_int("submit_seq", self.submit_seq)
        _check_non_negative_int("arrival_ns", self.arrival_ns)
        _check_non_negative_int("finished_ns", self.finished_ns)
        _check_non_negative_int("computed_tokens", self.computed_tokens)

        if not self.status.terminal:
            raise ValueError(
                "completion event must be terminal"
            )

        object.__setattr__(
            self,
            "output_token_ids",
            tuple(self.output_token_ids),
        )

        for name in ("admitted_ns", "first_token_ns"):
            value = getattr(self, name)
            if value is not None:
                _check_non_negative_int(name, value)


    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @classmethod
    def from_request(
        cls,
        request: RequestState,
    ) -> "CompletionEvent":
        if not request.terminal:
            raise ValueError(
                "request is not terminal"
            )

        if request.finished_ns is None:
            raise ValueError(
                "terminal request must have finished_ns"
            )

        return cls(
            request_id=request.request_id,
            status=request.status,
            output_token_ids=tuple(
                request.output_token_ids
            ),
            prompt_token_count=len(
                request.prompt_token_ids
            ),
            submit_seq=request.submit_seq,
            arrival_ns=request.arrival_ns,
            admitted_ns=request.admitted_ns,
            first_token_ns=request.first_token_ns,
            finished_ns=request.finished_ns,
            computed_tokens=request.computed_tokens,
        )
