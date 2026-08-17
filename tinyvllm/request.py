from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

from tinyvllm._validation import (
    _check_non_negative_int,
    _check_token_id,
)


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
