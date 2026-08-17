from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from tinyvllm.request import (
    RequestState,
    RequestStatus,
    SamplingParams,
    WorkClass,
)

from tinyvllm._validation import (
    _check_non_negative_int,
    _check_positive_int,
    _check_token_id,
)


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


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    plan_id: int
    mode: StepMode
    num_decode_items: int

    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    token_request_slots: tuple[int, ...]

    q_offsets: tuple[int, ...]
    context_lens_before: tuple[int, ...]
    query_lens: tuple[int, ...]
    context_lens_after: tuple[int, ...]

    sample_token_indices: tuple[int, ...]
    sample_request_ids: tuple[int, ...]
    sampling_params: tuple[SamplingParams, ...]

    def __post_init__(self) -> None:
        _check_non_negative_int("plan_id", self.plan_id)
        _check_non_negative_int(
            "num_decode_items",
            self.num_decode_items,
        )
        if not isinstance(self.mode, StepMode):
            raise TypeError("mode must be StepMode")

        tuple_fields = (
            "input_ids",
            "positions",
            "token_request_slots",
            "q_offsets",
            "context_lens_before",
            "query_lens",
            "context_lens_after",
            "sample_token_indices",
            "sample_request_ids",
            "sampling_params",
        )
        for name in tuple_fields:
            object.__setattr__(
                self,
                name,
                tuple(getattr(self, name)),
            )

        num_items = len(self.query_lens)
        if num_items == 0:
            raise ValueError(
                "PreparedBatch must contain at least one item"
            )
        if self.num_decode_items > num_items:
            raise ValueError(
                "num_decode_items exceeds item count"
            )

        expected_offsets = [0]
        for query_len in self.query_lens:
            _check_positive_int("query_len", query_len)
            expected_offsets.append(
                expected_offsets[-1] + query_len
            )

        if self.q_offsets != tuple(expected_offsets):
            raise ValueError(
                "q_offsets do not match query_lens"
            )

        num_query_tokens = expected_offsets[-1]
        for name in (
            "input_ids",
            "positions",
            "token_request_slots",
        ):
            if len(getattr(self, name)) != num_query_tokens:
                raise ValueError(
                    f"{name} length does not match query tokens"
                )

        for name in (
            "context_lens_before",
            "context_lens_after",
        ):
            if len(getattr(self, name)) != num_items:
                raise ValueError(
                    f"{name} length does not match item count"
                )

        for before, query_len, after in zip(
            self.context_lens_before,
            self.query_lens,
            self.context_lens_after,
        ):
            if after != before + query_len:
                raise ValueError(
                    "context lengths are inconsistent"
                )

        if any(
            query_len != 1
            for query_len in self.query_lens[
                : self.num_decode_items
            ]
        ):
            raise ValueError(
                "decode items must schedule exactly one token"
            )

        if self.num_decode_items == 0:
            expected_mode = StepMode.PREFILL_ONLY
        elif self.num_decode_items == num_items:
            expected_mode = StepMode.DECODE_ONLY
        else:
            expected_mode = StepMode.MIXED

        if self.mode is not expected_mode:
            raise ValueError(
                "mode does not match decode item count"
            )

        sample_count = len(self.sample_token_indices)
        if (
            len(self.sample_request_ids) != sample_count
            or len(self.sampling_params) != sample_count
        ):
            raise ValueError(
                "sampling metadata lengths do not match"
            )

        valid_sample_rows = {
            self.q_offsets[index + 1] - 1
            for index in range(num_items)
        }
        if any(
            index not in valid_sample_rows
            for index in self.sample_token_indices
        ):
            raise ValueError(
                "sample index must select an item's final query row"
            )

        if tuple(sorted(set(self.sample_token_indices))) != (
            self.sample_token_indices
        ):
            raise ValueError(
                "sample indices must be unique and ordered"
            )

        for params in self.sampling_params:
            if not isinstance(params, SamplingParams):
                raise TypeError(
                    "sampling_params must contain SamplingParams"
                )


    @property
    def num_items(self) -> int:
        return len(self.query_lens)

    @property
    def num_query_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def num_samples(self) -> int:
        return len(self.sample_token_indices)

    @property
    def num_decode_tokens(self) -> int:
        return self.q_offsets[self.num_decode_items]


@dataclass(frozen=True, slots=True)
class StepResult:
    """Ordered executor output for one dispatched plan."""

    plan_id: int
    sample_request_ids: tuple[int, ...]
    sampled_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _check_non_negative_int("plan_id", self.plan_id)

        object.__setattr__(
            self,
            "sample_request_ids",
            tuple(self.sample_request_ids),
        )
        object.__setattr__(
            self,
            "sampled_token_ids",
            tuple(self.sampled_token_ids),
        )

        if len(self.sample_request_ids) != len(self.sampled_token_ids):
            raise ValueError(
                "sample request and token counts do not match"
            )

        seen_request_ids: set[int] = set()
        for request_id in self.sample_request_ids:
            _check_non_negative_int("sample_request_id", request_id)
            if request_id in seen_request_ids:
                raise ValueError(
                    "sample_request_ids must be unique"
                )
            seen_request_ids.add(request_id)

        for token_id in self.sampled_token_ids:
            _check_token_id(token_id)


class BatchBuilder:
    def __init__(self, *, max_model_len: int) -> None:
        _check_positive_int("max_model_len", max_model_len)
        self.max_model_len = max_model_len


    def prepare(
        self,
        plan: StepPlan,
        requests: tuple[RequestState, ...]
    ) -> PreparedBatch:
        if not isinstance(plan, StepPlan):
            raise TypeError("plan must be StepPlan")

        requests = tuple(requests)
        if len(requests) != len(plan.items):
            raise ValueError(
                "requests must align with plan items"
            )

        input_ids: list[int] = []
        positions: list[int] = []
        token_request_slots: list[int] = []

        q_offsets = [0]
        context_lens_before: list[int] = []
        query_lens: list[int] = []
        context_lens_after: list[int] = []

        sample_token_indices: list[int] = []
        sample_request_ids: list[int] = []
        sampling_params: list[SamplingParams] = []

        for item, request in zip(plan.items, requests):
            if not isinstance(request, RequestState):
                raise TypeError(
                    "requests must contain RequestState values"
                )

            request.check_committed_invariants()

            if request.request_id != item.request_id:
                raise ValueError(
                    "request order does not match plan"
                )
            if request.status not in (
                RequestStatus.WAITING,
                RequestStatus.RUNNING,
            ):
                raise ValueError(
                    "planned request is not schedulable"
                )
            if request.inflight_plan_id is not None:
                raise ValueError(
                    "plan contains an inflight request"
                )
            if request.computed_tokens != item.token_start:
                raise ValueError(
                    "token_start does not match committed progress"
                )
            if item.context_len_before != request.computed_tokens:
                raise ValueError(
                    "context length does not match committed progress"
                )
            if request.work_class is not item.work_class:
                raise ValueError(
                    "work class does not match request"
                )
            if (
                request.active_slot is not None
                and request.active_slot != item.request_slot
            ):
                raise ValueError(
                    "request slot does not match active slot"
                )
            if item.context_len_after > self.max_model_len:
                raise ValueError(
                    "planned context exceeds max_model_len"
                )

            expected_sample = (
                item.token_end == request.total_tokens
            )
            if item.should_sample is not expected_sample:
                raise ValueError(
                    "should_sample does not match token interval"
                )

            tokens = request.token_slice(
                item.token_start,
                item.token_end,
            )
            if len(tokens) != item.query_len:
                raise RuntimeError(
                    "token slice length does not match query_len"
                )

            input_ids.extend(tokens)
            positions.extend(
                range(
                    item.context_len_before,
                    item.context_len_after,
                )
            )
            token_request_slots.extend(
                [item.request_slot] * item.query_len
            )

            q_offsets.append(len(input_ids))
            context_lens_before.append(
                item.context_len_before
            )
            query_lens.append(item.query_len)
            context_lens_after.append(
                item.context_len_after
            )

            if item.should_sample:
                sample_token_indices.append(
                    q_offsets[-1] - 1
                )
                sample_request_ids.append(item.request_id)
                sampling_params.append(request.sampling)

        return PreparedBatch(
            plan_id=plan.plan_id,
            mode=plan.mode,
            num_decode_items=plan.num_decode_items,
            input_ids=tuple(input_ids),
            positions=tuple(positions),
            token_request_slots=tuple(token_request_slots),
            q_offsets=tuple(q_offsets),
            context_lens_before=tuple(context_lens_before),
            query_lens=tuple(query_lens),
            context_lens_after=tuple(context_lens_after),
            sample_token_indices=tuple(sample_token_indices),
            sample_request_ids=tuple(sample_request_ids),
            sampling_params=tuple(sampling_params),
        )
