from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from tinyvllm.request import (
    RequestState,
    SamplingParams,
    WorkClass,
)

from tinyvllm._validation import _check_positive_int

from tinyvllm.cache import KvPlanTxn

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
        items = tuple(self.items)
        object.__setattr__(self, "items", items)

        if not items:
            raise ValueError("StepPlan must contain at least one item")

        seen_prefill = False
        num_decode_items = 0
        num_query_tokens = 0
        num_samples = 0

        for item in items:
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

            num_query_tokens += item.query_len
            num_samples += item.should_sample

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
        object.__setattr__(self, "num_query_tokens", num_query_tokens)
        object.__setattr__(self, "num_samples", num_samples)


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

    kv_page_table_ids: tuple[int, ...] = ()
    kv_page_table_offsets: tuple[int, ...] = ()
    kv_slot_mapping: tuple[int, ...] = ()


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

    @property
    def has_kv_metadata(self) -> bool:
        return bool(self.kv_page_table_offsets)


@dataclass(frozen=True, slots=True)
class StepResult:
    """Ordered executor output for one dispatched plan."""

    plan_id: int
    sample_request_ids: tuple[int, ...]
    sampled_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
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


class BatchBuilder:
    def __init__(self, *, max_model_len: int) -> None:
        _check_positive_int("max_model_len", max_model_len)
        self.max_model_len = max_model_len


    def prepare(
        self,
        plan: StepPlan,
        requests: tuple[RequestState, ...],
        *,
        kv: KvPlanTxn | None = None,
    ) -> PreparedBatch:
        if len(requests) != len(plan.items):
            raise ValueError(
                "requests must align with plan items"
            )

        if kv is not None and kv.plan_id != plan.plan_id:
            raise ValueError(
                "KV plan_id does not match step plan"
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

        kv_page_table_ids: list[int] = []
        kv_page_table_offsets = [0] if kv is not None else []
        kv_slot_mapping: list[int] = []

        for item, request in zip(plan.items, requests):
            if request.request_id != item.request_id:
                raise ValueError(
                    "request order does not match plan"
                )
            context_len_after = item.context_len_after
            if context_len_after > self.max_model_len:
                raise ValueError(
                    "planned context exceeds max_model_len"
                )

            tokens = request.token_slice(
                item.token_start,
                item.token_end,
            )
            if len(tokens) != item.query_len:
                raise RuntimeError(
                    "token slice length does not match query_len"
                )

            if kv is not None:
                page_table = kv.staged_page_table(
                    item.request_id
                )
                slot_mapping = kv.slot_mapping(
                    item.request_id
                )

                if len(slot_mapping) != item.query_len:
                    raise RuntimeError(
                        "KV slot mapping length does not match query_len"
                    )

                kv_page_table_ids.extend(
                    ref.page_id for ref in page_table
                )
                kv_page_table_offsets.append(
                    len(kv_page_table_ids)
                )
                kv_slot_mapping.extend(slot_mapping)

            input_ids.extend(tokens)
            positions.extend(
                range(
                    item.context_len_before,
                    context_len_after,
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
                context_len_after
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
            kv_page_table_ids=tuple(kv_page_table_ids),
            kv_page_table_offsets=tuple(kv_page_table_offsets),
            kv_slot_mapping=tuple(kv_slot_mapping),
        )
