from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from tinyvllm.request import WorkClass


def _check_non_negative_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


class StepMode(Enum):
    PREFILL_ONLY = auto()
    DECODE_ONLY = auto()
    MIXED = auto()


@dataclass(frozen=True, slots=True)
class WorkItem:
    request_id: int
    request_slot: int
    work_class: WorkClass
    token_start: int
    query_len: int
    context_len_before: int
    should_sample: bool


    def __post_init__(self) -> None:
        _check_non_negative_int("request_id", self.request_id)
        _check_non_negative_int("request_slot", self.request_slot)
        _check_non_negative_int("token_start", self.token_start)
        _check_non_negative_int(
            "context_len_before",
            self.context_len_before,
        )

        if type(self.query_len) is not int:
            raise TypeError("query_len must be an integer")
        if self.query_len <= 0:
            raise ValueError("query_len must be positive")
        if not isinstance(self.work_class, WorkClass):
            raise TypeError("work_class must be WorkClass")
        if type(self.should_sample) is not bool:
            raise TypeError("should_sample must be bool")


    @property
    def token_end(self) -> int:
        return self.token_start + self.query_len


    @property
    def context_len_after(self) -> int:
        return self.context_len_before + self.query_len


@dataclass(frozen=True, slots=True)
class StepPlan:
    plan_id: int
    items: tuple[WorkItem, ...]

    mode: StepMode = field(init=False)
    num_decode_items: int = field(init=False)
    num_prefill_items: int = field(init=False)
    num_query_tokens: int = field(init=False)
    num_samples: int = field(init=False)


    def __post_init__(self) -> None:
        _check_non_negative_int("plan_id", self.plan_id)

        items = tuple(self.items)
        object.__setattr__(self, "items", items)

        if not items:
            raise ValueError("StepPlan must contain at least one item")

        request_ids: set[int] = set()
        request_slots: set[int] = set()
        seen_prefill = False
        num_decode_items = 0

        for item in items:
            if not isinstance(item, WorkItem):
                raise TypeError("items must contain WorkItem values")

            if item.request_id in request_ids:
                raise ValueError(
                    "a request may appear only once in a StepPlan"
                )
            if item.request_slot in request_slots:
                raise ValueError(
                    "a request slot may appear only once in a StepPlan"
                )

            request_ids.add(item.request_id)
            request_slots.add(item.request_slot)

            if item.work_class is WorkClass.PREFILL:
                seen_prefill = True
            elif item.work_class is WorkClass.DECODE:
                if seen_prefill:
                    raise ValueError(
                        "decode items must precede prefill items"
                    )
                num_decode_items += 1
            else:
                raise ValueError(
                    f"unsupported work class: {item.work_class}"
                )

        num_prefill_items = len(items) - num_decode_items

        if num_decode_items == 0:
            mode = StepMode.PREFILL_ONLY
        elif num_prefill_items == 0:
            mode = StepMode.DECODE_ONLY
        else:
            mode = StepMode.MIXED

        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "num_decode_items", num_decode_items)
        object.__setattr__(self, "num_prefill_items", num_prefill_items)
        object.__setattr__(self, "num_query_tokens", sum(item.query_len for item in items))
        object.__setattr__(self, "num_samples", sum(item.should_sample for item in items))
            