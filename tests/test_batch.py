from __future__ import annotations

import pytest

import copy

from tinyvllm.batch import (
    BatchBuilder,
    StepResult,
    StepMode,
    StepPlan,
    WorkItem,
)
from tinyvllm.executor import CpuFakeExecutor
from tinyvllm.request import (
    RequestState,
    RequestStatus,
    SamplingParams,
    WorkClass,
)

from tinyvllm.cache import (
    KvAppend,
    KvPageManager,
    KvTxnState,
)


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

def waiting_request(
    request_id: int = 1,
    *,
    prompt: tuple[int, ...] = (21, 22, 23),
) -> RequestState:
    return RequestState(
        request_id=request_id,
        submit_seq=request_id,
        prompt_token_ids=list(prompt),
        sampling=SamplingParams(max_new_tokens=4),
        arrival_ns=0,
        status=RequestStatus.WAITING,
    )


def running_decode_request() -> RequestState:
    return RequestState(
        request_id=2,
        submit_seq=2,
        prompt_token_ids=[10, 11, 12],
        sampling=SamplingParams(max_new_tokens=4),
        arrival_ns=0,
        output_token_ids=[13],
        status=RequestStatus.RUNNING,
        work_class=WorkClass.DECODE,
        computed_tokens=3,
        active_slot=1,
        admitted_ns=0,
        first_token_ns=1,
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


def test_plan_rejects_empty_items() -> None:
    with pytest.raises(ValueError, match="at least one"):
        StepPlan(plan_id=1, items=())


def test_builder_flattens_mixed_batch_without_padding() -> None:
    decode = running_decode_request()
    prefill = waiting_request(
        prompt=(21, 22, 23, 24, 25)
    )
    before = copy.deepcopy((decode, prefill))

    plan = StepPlan(
        plan_id=7,
        items=(
            decode_item(),
            prefill_item(should_sample=False),
        ),
    )

    batch = BatchBuilder(max_model_len=16).prepare(
        plan,
        (decode, prefill),
    )

    assert batch.mode is StepMode.MIXED
    assert batch.input_ids == (13, 21, 22, 23)
    assert batch.positions == (3, 0, 1, 2)
    assert batch.token_request_slots == (1, 0, 0, 0)
    assert batch.q_offsets == (0, 1, 4)
    assert batch.context_lens_before == (3, 0)
    assert batch.query_lens == (1, 3)
    assert batch.context_lens_after == (4, 3)
    assert batch.num_decode_tokens == 1

    # Partial prefill does not produce a sampling row.
    assert batch.sample_token_indices == (0,)
    assert batch.sample_request_ids == (2,)
    assert batch.sampling_params == (decode.sampling,)

    assert (decode, prefill) == before


def test_final_prefill_maps_its_last_query_row() -> None:
    request = waiting_request()
    plan = StepPlan(
        plan_id=8,
        items=(prefill_item(),),
    )

    batch = BatchBuilder(max_model_len=16).prepare(
        plan,
        (request,),
    )

    assert batch.sample_token_indices == (2,)
    assert batch.sample_request_ids == (1,)
    assert batch.num_samples == 1


def test_builder_rejects_wrong_request_order() -> None:
    plan = StepPlan(
        plan_id=10,
        items=(decode_item(), prefill_item()),
    )

    with pytest.raises(ValueError, match="request order"):
        BatchBuilder(max_model_len=16).prepare(
            plan,
            (
                waiting_request(),
                running_decode_request(),
            ),
        )


def test_builder_rejects_context_over_model_limit() -> None:
    request = waiting_request()

    with pytest.raises(ValueError, match="max_model_len"):
        BatchBuilder(max_model_len=2).prepare(
            StepPlan(
                plan_id=11,
                items=(prefill_item(),),
            ),
            (request,),
        )


def test_step_result_copies_and_validates_parallel_arrays() -> None:
    request_ids = [10, 11]
    token_ids = [101, 102]

    result = StepResult(
        plan_id=12,
        sample_request_ids=request_ids,
        sampled_token_ids=token_ids,
    )

    request_ids.append(12)
    token_ids.append(103)

    assert result.sample_request_ids == (10, 11)
    assert result.sampled_token_ids == (101, 102)

    with pytest.raises(ValueError, match="counts"):
        StepResult(12, (10,), (101, 102))


def test_fake_executor_preserves_sampling_order() -> None:
    request = waiting_request()

    batch = BatchBuilder(max_model_len=16).prepare(
        StepPlan(
            plan_id=13,
            items=(prefill_item(),),
        ),
        (request,),
    )

    result = CpuFakeExecutor(token_id=99).execute(batch)

    assert result == StepResult(
        plan_id=13,
        sample_request_ids=(1,),
        sampled_token_ids=(99,),
    )


def test_fake_executor_returns_empty_partial_prefill_result() -> None:
    request = waiting_request(
        prompt=(21, 22, 23, 24, 25),
    )

    batch = BatchBuilder(max_model_len=16).prepare(
        StepPlan(
            plan_id=14,
            items=(prefill_item(should_sample=False),),
        ),
        (request,),
    )

    result = CpuFakeExecutor(token_id=99).execute(batch)

    assert result.sample_request_ids == ()
    assert result.sampled_token_ids == ()


def test_builder_carries_staged_kv_metadata_in_plan_order() -> None:
    manager = KvPageManager(page_size=4, num_pages=4)

    initial = manager.begin_plan(
        plan_id=1,
        appends=(KvAppend(2, 0, 3),),
    )
    assert initial is not None
    initial.commit_before_dispatch()
    initial.complete_writes()

    decode = running_decode_request()
    prefill = waiting_request(
        request_id=1,
        prompt=(21, 22, 23),
    )

    plan = StepPlan(
        plan_id=2,
        items=(decode_item(), prefill_item()),
    )

    txn = manager.begin_plan(
        plan_id=plan.plan_id,
        appends=(
            KvAppend(2, 3, 1),
            KvAppend(1, 0, 3),
        ),
    )
    assert txn is not None

    batch = BatchBuilder(max_model_len=16).prepare(
        plan,
        (decode, prefill),
        kv=txn,
    )

    assert batch.has_kv_metadata
    assert batch.kv_page_table_ids == (0, 1)
    assert batch.kv_page_table_offsets == (0, 1, 2)
    assert batch.kv_slot_mapping == (3, 4, 5, 6)

    for index, expected in enumerate(((0,), (1,))):
        start = batch.kv_page_table_offsets[index]
        end = batch.kv_page_table_offsets[index + 1]
        assert batch.kv_page_table_ids[start:end] == expected

    txn.rollback()


def test_builder_rejects_kv_metadata_from_another_plan() -> None:
    manager = KvPageManager(page_size=4, num_pages=2)

    txn = manager.begin_plan(
        plan_id=20,
        appends=(KvAppend(1, 0, 1),),
    )
    assert txn is not None

    with pytest.raises(ValueError, match="plan_id"):
        BatchBuilder(max_model_len=16).prepare(
            StepPlan(
                plan_id=21,
                items=(prefill_item(),),
            ),
            (waiting_request(),),
            kv=txn,
        )

    txn.rollback()


def test_builder_prepare_keeps_kv_transaction_read_only() -> None:
    manager = KvPageManager(page_size=4, num_pages=3)

    txn = manager.begin_plan(
        plan_id=30,
        appends=(KvAppend(1, 0, 3),),
    )
    assert txn is not None

    before_allocator = manager.allocator.snapshot()
    before_table = manager.page_table(1)
    before_valid = manager.valid_token_count(1)

    batch = BatchBuilder(max_model_len=16).prepare(
        StepPlan(
            plan_id=30,
            items=(prefill_item(request_id=1, slot=0),),
        ),
        (
            waiting_request(
                request_id=1,
                prompt=(21, 22, 23),
            ),
        ),
        kv=txn,
    )

    assert batch.has_kv_metadata
    assert txn.state is KvTxnState.PREPARED
    assert manager.allocator.snapshot() == before_allocator
    assert manager.page_table(1) == before_table
    assert manager.valid_token_count(1) == before_valid

    txn.rollback()


def test_builder_keeps_partial_page_slots_when_page_table_grows() -> None:
    manager = KvPageManager(page_size=4, num_pages=4)

    first = manager.begin_plan(
        plan_id=40,
        appends=(KvAppend(1, 0, 3),),
    )
    assert first is not None
    first.commit_before_dispatch()
    first.complete_writes()

    second = manager.begin_plan(
        plan_id=41,
        appends=(KvAppend(1, 3, 3),),
    )
    assert second is not None

    request = waiting_request(
        request_id=1,
        prompt=(21, 22, 23, 24, 25, 26),
    )

    plan = StepPlan(
        plan_id=41,
        items=(
            WorkItem(
                request_id=1,
                request_slot=0,
                work_class=WorkClass.PREFILL,
                token_start=3,
                query_len=3,
                context_len_before=3,
                should_sample=True,
            ),
        ),
    )

    batch = BatchBuilder(max_model_len=16).prepare(
        plan,
        (request,),
        kv=second,
    )

    assert batch.kv_page_table_ids == (0, 1)
    assert batch.kv_page_table_offsets == (0, 2)
    assert batch.kv_slot_mapping == (3, 4, 5)

    second.rollback()