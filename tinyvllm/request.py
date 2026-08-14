from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto


def _check_non_negative_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


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