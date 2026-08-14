from __future__ import annotations

import pytest

from tinyvllm.batch import StepMode, StepPlan, WorkItem
from tinyvllm.request import WorkClass


def prefill_item(
    request_id: int = 1,
    slot: int = 0,
    *,
    should_sample: bool = True,
) -> WorkItem:
    return WorkItem(
        request_id=request_id,
        request_slot=slot,
        work_class=WorkClass.PREFILL,
        token_start=0,
        query_len=3,
        context_len_before=0,
        should_sample=should_sample,
    )


def decode_item(
    request_id: int = 2,
    slot: int = 1,
) -> WorkItem:
    return WorkItem(
        request_id=request_id,
        request_slot=slot,
        work_class=WorkClass.DECODE,
        token_start=3,
        query_len=1,
        context_len_before=3,
        should_sample=True,
    )


def test_work_item_lengths() -> None:
    item = prefill_item()

    assert item.token_end == 3
    assert item.context_len_after == 3


def test_prefill_only_plan() -> None:
    plan = StepPlan(1, (prefill_item(),))

    assert plan.mode is StepMode.PREFILL_ONLY
    assert plan.num_decode_items == 0
    assert plan.num_prefill_items == 1
    assert plan.num_query_tokens == 3
    assert plan.num_samples == 1


def test_decode_only_plan() -> None:
    plan = StepPlan(1, (decode_item(),))

    assert plan.mode is StepMode.DECODE_ONLY
    assert plan.num_decode_items == 1


def test_mixed_plan_is_decode_first() -> None:
    plan = StepPlan(
        plan_id=1,
        items=(decode_item(), prefill_item()),
    )

    assert plan.mode is StepMode.MIXED
    assert plan.num_query_tokens == 4
    assert plan.num_samples == 2


def test_partial_prefill_has_no_sample() -> None:
    plan = StepPlan(
        plan_id=1,
        items=(prefill_item(should_sample=False),),
    )

    assert plan.mode is StepMode.PREFILL_ONLY
    assert plan.num_samples == 0


def test_plan_rejects_prefill_before_decode() -> None:
    with pytest.raises(ValueError, match="precede"):
        StepPlan(
            plan_id=1,
            items=(prefill_item(), decode_item()),
        )


def test_plan_rejects_duplicate_request() -> None:
    with pytest.raises(ValueError, match="request may appear"):
        StepPlan(
            plan_id=1,
            items=(prefill_item(), prefill_item(slot=1)),
        )


def test_plan_rejects_empty_items() -> None:
    with pytest.raises(ValueError, match="at least one"):
        StepPlan(plan_id=1, items=())