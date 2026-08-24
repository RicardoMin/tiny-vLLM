from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import pytest

from tinyvllm.request import (
    RequestStatus,
    SamplingParams,
    WorkClass
)
from tinyvllm.scheduler import AddCommand, Scheduler, TxnState
from tinyvllm.batch import (
    BatchBuilder,
    StepResult,
    StepMode,
)
from tinyvllm.executor import CpuFakeExecutor
from tinyvllm.cache import KvTxnState


def command(
    seq: int,
    request_id: int,
    *,
    arrival_ns: int = 0,
    prompt_len: int = 3,
    max_new_tokens: int = 4,
) -> AddCommand:
    return AddCommand(
        seq=seq,
        request_id=request_id,
        prompt_token_ids=tuple(range(1, prompt_len + 1)),
        sampling=SamplingParams(
            max_new_tokens=max_new_tokens
        ),
        arrival_ns=arrival_ns,
    )


def make_scheduler(
    *,
    active: int = 4,
    rows: int | None = None,
    tokens: int = 8,
    chunk: int = 4,
    model_len: int = 32,
    reserve_tokens: int = 128,
    reserve_rows: int = 1,
    kv_page_size: int = 16,
    kv_num_pages: int | None = None,
) -> Scheduler:
    return Scheduler(
        max_active_requests=active,
        max_step_requests=active if rows is None else rows,
        max_scheduled_tokens=tokens,
        prefill_chunk_size=chunk,
        max_model_len=model_len,
        prefill_reserve_tokens=reserve_tokens,
        prefill_reserve_requests=reserve_rows,
        kv_page_size=kv_page_size,
        kv_num_pages=kv_num_pages,
    )


def committed_state(scheduler: Scheduler):
    return (
        scheduler.waiting_request_ids,
        tuple(
            (
                request_id,
                copy.deepcopy(
                    scheduler.get_request(request_id)
                ),
            )
            for request_id in scheduler.request_ids
        ),
    )


def test_command_cutoff_is_external() -> None:
    scheduler = make_scheduler()

    scheduler.apply_commands(
        (command(0, 10),),
        now_ns=0,
    )

    assert scheduler.request_ids == (10,)

    scheduler.apply_commands(
        (command(1, 11),),
        now_ns=0,
    )

    assert scheduler.request_ids == (10, 11)


def test_command_batch_is_atomic() -> None:
    scheduler = make_scheduler()

    with pytest.raises(
        ValueError,
        match="expected command seq 1",
    ):
        scheduler.apply_commands(
            (
                command(0, 10),
                command(2, 11),
            ),
            now_ns=0,
        )

    assert scheduler.request_ids == ()
    assert scheduler.last_command_seq == -1


def test_prompt_over_model_limit_is_rejected_atomically() -> None:
    scheduler = make_scheduler(model_len=3)

    with pytest.raises(ValueError, match="prompt exceeds"):
        scheduler.apply_commands(
            (command(0, 10, prompt_len=4),),
            now_ns=0,
        )

    assert scheduler.request_ids == ()
    assert scheduler.last_command_seq == -1


def test_scheduler_requires_matching_prefill_reservations() -> None:
    with pytest.raises(ValueError, match="both be zero or positive"):
        make_scheduler(reserve_tokens=1, reserve_rows=0)


def test_future_arrival_stays_pending() -> None:
    scheduler = make_scheduler()

    scheduler.apply_commands(
        (command(0, 10, arrival_ns=100),),
        now_ns=50,
    )

    assert (
        scheduler.get_request(10).status
        is RequestStatus.PENDING
    )
    assert scheduler.waiting_request_ids == ()

    scheduler.apply_commands((), now_ns=100)

    assert (
        scheduler.get_request(10).status
        is RequestStatus.WAITING
    )
    assert scheduler.waiting_request_ids == (10,)


def test_future_head_does_not_block_eligible_request() -> None:
    scheduler = make_scheduler()

    scheduler.apply_commands(
        (
            command(0, 10, arrival_ns=100),
            command(1, 11, arrival_ns=10),
        ),
        now_ns=10,
    )

    assert scheduler.waiting_request_ids == (11,)


def test_new_arrivals_use_fifo_order() -> None:
    scheduler = make_scheduler()

    scheduler.apply_commands(
        (
            command(0, 10, arrival_ns=20),
            command(1, 11, arrival_ns=10),
            command(2, 12, arrival_ns=10),
        ),
        now_ns=20,
    )

    assert scheduler.waiting_request_ids == (
        11,
        12,
        10,
    )


def test_zero_output_finishes_without_gpu() -> None:
    scheduler = make_scheduler()

    scheduler.apply_commands(
        (command(0, 10, max_new_tokens=0),),
        now_ns=0,
    )

    request = scheduler.get_request(10)

    assert request.status is RequestStatus.FINISHED
    assert request.admitted_ns == 0
    assert request.finished_ns == 0

    assert (
        scheduler.take_immediate_finished_request_ids()
        == (10,)
    )
    assert (
        scheduler.take_immediate_finished_request_ids()
        == ()
    )
    assert scheduler.begin_step(now_ns=0) is None


def test_clock_cannot_move_backwards() -> None:
    scheduler = make_scheduler()

    scheduler.apply_commands(
        (command(0, 10),),
        now_ns=10,
    )

    with pytest.raises(ValueError, match="backwards"):
        scheduler.apply_commands((), now_ns=9)


def test_add_command_copies_prompt_tokens() -> None:
    prompt = [1, 2, 3]

    add = AddCommand(
        seq=0,
        request_id=10,
        prompt_token_ids=prompt,
        sampling=SamplingParams(),
        arrival_ns=0,
    )
    prompt.append(4)

    assert add.prompt_token_ids == (1, 2, 3)


def test_prepare_is_pure_and_uses_lowest_slots() -> None:
    scheduler = make_scheduler(active=2)
    scheduler.apply_commands(
        (command(0, 10), command(1, 11), command(2, 12)),
        now_ns=0,
    )
    before = committed_state(scheduler)

    txn = scheduler.begin_step(now_ns=0)

    assert txn is not None
    assert [
        (item.request_id, item.request_slot)
        for item in txn.plan.items
    ] == [(10, 0), (11, 1)]
    assert committed_state(scheduler) == before

    scheduler.rollback(txn)
    assert committed_state(scheduler) == before


def test_token_budget_and_chunk_size_are_enforced() -> None:
    scheduler = make_scheduler(
        active=3,
        tokens=4,
        chunk=3,
    )
    scheduler.apply_commands(
        (
            command(0, 10, prompt_len=5),
            command(1, 11, prompt_len=5),
        ),
        now_ns=0,
    )

    txn = scheduler.begin_step(now_ns=0)

    assert txn is not None
    assert [item.query_len for item in txn.plan.items] == [3, 1]
    assert txn.plan.num_query_tokens == 4
    assert txn.plan.num_samples == 0
    scheduler.rollback(txn)


def test_only_one_transaction_may_be_open() -> None:
    scheduler = make_scheduler()
    scheduler.apply_commands((command(0, 10),), now_ns=0)

    txn = scheduler.begin_step(now_ns=0)

    assert txn is not None
    assert scheduler.has_open_transaction

    with pytest.raises(RuntimeError, match="already open"):
        scheduler.begin_step(now_ns=0)

    with pytest.raises(RuntimeError, match="open transaction"):
        scheduler.apply_commands((), now_ns=0)

    scheduler.rollback(txn)
    assert txn.state is TxnState.ROLLED_BACK
    assert not scheduler.has_open_transaction


def test_schedule_transaction_is_immutable() -> None:
    scheduler = make_scheduler()
    scheduler.apply_commands((command(0, 10),), now_ns=0)
    txn = scheduler.begin_step(now_ns=0)
    assert txn is not None

    with pytest.raises(FrozenInstanceError):
        txn.plan = txn.plan
    with pytest.raises(FrozenInstanceError):
        txn.state = TxnState.COMMITTED

    scheduler.rollback(txn)


def test_rollback_retry_uses_a_fresh_plan_id() -> None:
    scheduler = make_scheduler()
    scheduler.apply_commands((command(0, 10),), now_ns=0)

    first = scheduler.begin_step(now_ns=0)
    assert first is not None
    scheduler.rollback(first)

    second = scheduler.begin_step(now_ns=0)
    assert second is not None
    assert second.plan.plan_id == first.plan.plan_id + 1
    scheduler.rollback(second)


def test_begin_step_requires_same_command_cutoff_time() -> None:
    scheduler = make_scheduler()
    scheduler.apply_commands((command(0, 10),), now_ns=10)

    with pytest.raises(ValueError, match="same now_ns"):
        scheduler.begin_step(now_ns=11)


def test_mark_dispatched_publishes_runtime_ownership() -> None:
    scheduler = make_scheduler(active=2)
    scheduler.apply_commands(
        (
            command(0, 10),
            command(1, 11),
            command(2, 12),
            command(3, 13, arrival_ns=11),
        ),
        now_ns=10,
    )
    txn = scheduler.begin_step(now_ns=10)
    assert txn is not None

    batch = BatchBuilder(max_model_len=16).prepare(
        txn.plan,
        tuple(
            scheduler.get_request(item.request_id)
            for item in txn.plan.items
        ),
        kv=txn.kv,
    )
    assert batch.plan_id == txn.plan.plan_id

    scheduler.mark_dispatched(txn, now_ns=12)

    assert txn.state is TxnState.DISPATCHED
    assert txn.kv.state is KvTxnState.DISPATCHED
    assert scheduler.kv_manager.page_table(10)
    assert scheduler.kv_manager.valid_token_count(10) == 0
    assert scheduler.has_open_transaction
    assert scheduler.waiting_request_ids == (12,)
    assert (
        scheduler.get_request(13).status
        is RequestStatus.PENDING
    )

    for request_id, slot in ((10, 0), (11, 1)):
        request = scheduler.get_request(request_id)
        assert request.status is RequestStatus.RUNNING
        assert request.active_slot == slot
        assert request.admitted_ns == 12
        assert request.inflight_plan_id == txn.plan.plan_id
        assert request.computed_tokens == 0
        assert request.output_token_ids == []


def test_dispatched_transaction_cannot_roll_back() -> None:
    scheduler = make_scheduler()
    scheduler.apply_commands((command(0, 10),), now_ns=0)
    txn = scheduler.begin_step(now_ns=0)
    assert txn is not None

    scheduler.mark_dispatched(txn, now_ns=1)

    assert txn.kv.state is KvTxnState.DISPATCHED

    with pytest.raises(RuntimeError, match="must be prepared"):
        scheduler.rollback(txn)

    assert txn.state is TxnState.DISPATCHED
    assert scheduler.has_open_transaction
    assert (
        scheduler.get_request(10).inflight_plan_id
        == txn.plan.plan_id
    )


def _run_fake_step(scheduler: Scheduler):
    txn = scheduler.begin_step(now_ns=0)
    assert txn is not None

    batch = BatchBuilder(max_model_len=16).prepare(
        txn.plan,
        tuple(
            scheduler.get_request(item.request_id)
            for item in txn.plan.items
        ),
        kv=txn.kv,
    )
    scheduler.mark_dispatched(txn, now_ns=1)

    result = CpuFakeExecutor(token_id=99).execute(batch)
    return txn, result


@pytest.mark.parametrize(
    ("prompt_len", "chunk", "max_new_tokens", "expected"),
    (
        (
            5, 2, 4,
            (2, (), RequestStatus.RUNNING,
             WorkClass.PREFILL, 0, ()),
        ),
        (
            3, 4, 4,
            (3, (99,), RequestStatus.RUNNING,
             WorkClass.DECODE, 0, ()),
        ),
        (
            3, 4, 1,
            (3, (99,), RequestStatus.FINISHED,
             WorkClass.DECODE, None, (10,)),
        ),
    ),
)
def test_commit_request_lifecycle(
    prompt_len: int,
    chunk: int,
    max_new_tokens: int,
    expected: tuple,
) -> None:
    scheduler = make_scheduler(active=1, chunk=chunk)
    scheduler.apply_commands(
        (
            command(
                0,
                10,
                prompt_len=prompt_len,
                max_new_tokens=max_new_tokens,
            ),
        ),
        now_ns=0,
    )
    txn, result = _run_fake_step(scheduler)

    finished_ids = scheduler.commit_step(
        txn,
        result,
        now_ns=2,
    )
    request = scheduler.get_request(10)

    actual = (
        request.computed_tokens,
        tuple(request.output_token_ids),
        request.status,
        request.work_class,
        request.active_slot,
        finished_ids,
    )
    assert actual == expected

    output = expected[1]
    assert request.total_tokens == prompt_len + len(output)
    assert request.inflight_plan_id is None
    assert request.first_token_ns == (2 if output else None)
    assert request.finished_ns == (
        2 if finished_ids else None
    )
    assert txn.state is TxnState.COMMITTED
    assert txn.kv.state is KvTxnState.COMMITTED
    assert not scheduler.has_open_transaction

    if finished_ids:
        assert scheduler.kv_manager.page_table(10) == ()
        assert scheduler.kv_manager.valid_token_count(10) == 0
        scheduler.apply_commands(
            (command(1, 11),),
            now_ns=3,
        )
        next_txn = scheduler.begin_step(now_ns=3)
        assert next_txn is not None
        assert next_txn.plan.items[0].request_slot == 0
        scheduler.rollback(next_txn)


@pytest.mark.parametrize(
    "bad_result",
    ("plan", "order", "token"),
)
def test_bad_result_fails_before_publish(
    bad_result: str,
) -> None:
    scheduler = make_scheduler(active=2)
    scheduler.apply_commands(
        (command(0, 10), command(1, 11)),
        now_ns=0,
    )
    txn, result = _run_fake_step(scheduler)
    before = committed_state(scheduler)

    if bad_result == "plan":
        result = StepResult(
            result.plan_id + 1,
            (10, 11),
            (99, 99),
        )
    elif bad_result == "order":
        result = StepResult(
            result.plan_id,
            (11, 10),
            (99, 99),
        )
    else:
        result = StepResult(
            result.plan_id,
            (10, 11),
            (99, -1),
        )

    with pytest.raises((RuntimeError, ValueError)):
        scheduler.commit_step(txn, result, now_ns=2)

    assert committed_state(scheduler) == before
    assert txn.state is TxnState.FAILED
    assert txn.kv.state is KvTxnState.DISPATCHED
    assert scheduler.kv_manager.valid_token_count(10) == 0
    assert scheduler.has_open_transaction


def test_executor_failure_is_fail_stop() -> None:
    scheduler = make_scheduler()
    scheduler.apply_commands(
        (command(0, 10),),
        now_ns=0,
    )
    txn, _ = _run_fake_step(scheduler)

    scheduler.fail_step(txn)

    request = scheduler.get_request(10)
    assert txn.state is TxnState.FAILED
    assert txn.kv.state is KvTxnState.DISPATCHED
    assert request.inflight_plan_id == txn.plan.plan_id

    with pytest.raises(RuntimeError, match="open transaction"):
        scheduler.apply_commands((), now_ns=2)

    with pytest.raises(RuntimeError, match="already open"):
        scheduler.begin_step(now_ns=2)

    with pytest.raises(RuntimeError, match="must be prepared"):
        scheduler.rollback(txn)


def _commit_fake_step(scheduler: Scheduler, now_ns: int):
    scheduler.apply_commands((), now_ns=now_ns)
    txn = scheduler.begin_step(now_ns=now_ns)
    assert txn is not None

    batch = BatchBuilder(max_model_len=32).prepare(
        txn.plan,
        tuple(
            scheduler.get_request(item.request_id)
            for item in txn.plan.items
        ),
        kv=txn.kv,
    )
    scheduler.mark_dispatched(txn, now_ns=now_ns + 1)
    result = CpuFakeExecutor(token_id=99).execute(batch)
    finished_ids = scheduler.commit_step(
        txn,
        result,
        now_ns=now_ns + 2,
    )
    return txn.plan, finished_ids


def test_active_request_runs_until_completion() -> None:
    scheduler = make_scheduler(
        active=1,
        tokens=2,
        chunk=2,
    )
    scheduler.apply_commands(
        (
            command(
                0,
                10,
                prompt_len=5,
                max_new_tokens=3,
            ),
        ),
        now_ns=0,
    )

    observed = []
    for now_ns in range(0, 15, 3):
        plan, finished_ids = _commit_fake_step(
            scheduler,
            now_ns,
        )
        item = plan.items[0]
        observed.append(
            (
                item.work_class,
                item.query_len,
                item.should_sample,
            )
        )

    assert observed == [
        (WorkClass.PREFILL, 2, False),
        (WorkClass.PREFILL, 2, False),
        (WorkClass.PREFILL, 1, True),
        (WorkClass.DECODE, 1, True),
        (WorkClass.DECODE, 1, True),
    ]
    assert finished_ids == (10,)

    request = scheduler.get_request(10)
    assert (
        request.computed_tokens,
        request.total_tokens,
        request.output_token_ids,
        request.status,
        request.active_slot,
        request.first_token_ns,
        request.finished_ns,
    ) == (
        7,
        8,
        [99, 99, 99],
        RequestStatus.FINISHED,
        None,
        8,
        14,
    )

    scheduler.apply_commands((), now_ns=15)
    assert scheduler.begin_step(now_ns=15) is None


def test_model_limit_finishes_without_an_invalid_decode() -> None:
    scheduler = make_scheduler(
        active=1,
        tokens=4,
        chunk=4,
        model_len=3,
    )
    scheduler.apply_commands(
        (command(0, 10, prompt_len=3, max_new_tokens=4),),
        now_ns=0,
    )

    txn = scheduler.begin_step(now_ns=0)
    assert txn is not None
    batch = BatchBuilder(max_model_len=3).prepare(
        txn.plan,
        (scheduler.get_request(10),),
        kv=txn.kv,
    )
    scheduler.mark_dispatched(txn, now_ns=1)
    finished_ids = scheduler.commit_step(
        txn,
        CpuFakeExecutor(token_id=99).execute(batch),
        now_ns=2,
    )

    request = scheduler.get_request(10)
    assert finished_ids == (10,)
    assert request.status is RequestStatus.FINISHED
    assert request.computed_tokens == 3
    assert request.output_token_ids == [99]

    scheduler.apply_commands((), now_ns=3)
    assert scheduler.begin_step(now_ns=3) is None


def test_mixed_active_and_waiting_share_budgets() -> None:
    scheduler = make_scheduler(
        active=3,
        rows=3,
        tokens=4,
        chunk=3,
    )
    scheduler.apply_commands(
        (
            command(0, 10, prompt_len=1),
            command(1, 11, prompt_len=5),
            command(2, 12, prompt_len=5),
        ),
        now_ns=0,
    )
    first_plan, _ = _commit_fake_step(scheduler, 0)

    assert tuple(
        item.request_id for item in first_plan.items
    ) == (10, 11)
    assert scheduler.waiting_request_ids == (12,)

    scheduler.apply_commands((), now_ns=3)
    txn = scheduler.begin_step(now_ns=3)
    assert txn is not None

    assert tuple(
        (
            item.request_id,
            item.request_slot,
            item.work_class,
            item.query_len,
        )
        for item in txn.plan.items
    ) == (
        (10, 0, WorkClass.DECODE, 1),
        (11, 1, WorkClass.PREFILL, 2),
        (12, 2, WorkClass.PREFILL, 1),
    )
    assert txn.plan.mode is StepMode.MIXED
    assert txn.plan.num_query_tokens == 4

    batch = BatchBuilder(max_model_len=32).prepare(
        txn.plan,
        tuple(
            scheduler.get_request(item.request_id)
            for item in txn.plan.items
        ),
        kv=txn.kv,
    )
    scheduler.mark_dispatched(txn, now_ns=4)

    assert scheduler.waiting_request_ids == ()
    assert tuple(
        scheduler.get_request(request_id).active_slot
        for request_id in (10, 11, 12)
    ) == (0, 1, 2)

    result = CpuFakeExecutor(token_id=99).execute(batch)
    scheduler.commit_step(txn, result, now_ns=5)


def test_waiting_prefill_gets_a_turn_while_active_prefill_runs() -> None:
    scheduler = make_scheduler(
        active=5,
        rows=3,
        tokens=6,
        chunk=2,
        model_len=128,
        reserve_tokens=2,
        reserve_rows=1,
    )
    scheduler.apply_commands(
        (
            command(0, 10, prompt_len=1, max_new_tokens=100),
            command(1, 11, prompt_len=1, max_new_tokens=100),
            command(2, 12, prompt_len=100, max_new_tokens=2),
        ),
        now_ns=0,
    )
    first_plan, _ = _commit_fake_step(scheduler, 0)
    assert tuple(item.request_id for item in first_plan.items) == (
        10,
        11,
        12,
    )

    scheduler.apply_commands(
        (command(3, 13, prompt_len=1),),
        now_ns=3,
    )
    _commit_fake_step(scheduler, 3)
    third_plan, _ = _commit_fake_step(scheduler, 6)

    request_ids = tuple(item.request_id for item in third_plan.items)
    assert 13 in request_ids
    assert len(request_ids) == len(set(request_ids))
    assert scheduler.get_request(13).status is RequestStatus.RUNNING


def test_prefill_reservation_protects_progress() -> None:
    scheduler = make_scheduler(
        active=4,
        rows=3,
        tokens=6,
        chunk=4,
        reserve_tokens=2,
        reserve_rows=1,
    )
    scheduler.apply_commands(
        (
            command(0, 10, prompt_len=1),
            command(1, 11, prompt_len=1),
            command(2, 12, prompt_len=1),
            command(3, 13, prompt_len=5),
        ),
        now_ns=0,
    )

    first_plan, _ = _commit_fake_step(scheduler, 0)
    assert tuple(
        item.request_id for item in first_plan.items
    ) == (10, 11, 12)

    scheduler.apply_commands((), now_ns=3)
    txn = scheduler.begin_step(now_ns=3)
    assert txn is not None

    items = tuple(
        (
            item.request_id,
            item.work_class,
            item.query_len,
        )
        for item in txn.plan.items
    )
    assert items == (
        (10, WorkClass.DECODE, 1),
        (11, WorkClass.DECODE, 1),
        (13, WorkClass.PREFILL, 2),
    )
    assert len(
        {item.request_id for item in txn.plan.items}
    ) == len(txn.plan.items)

    scheduler.rollback(txn)


def test_unused_prefill_reservation_backfills_decode() -> None:
    scheduler = make_scheduler(
        active=5,
        rows=4,
        tokens=8,
        chunk=4,
        reserve_tokens=2,
        reserve_rows=2,
    )
    scheduler.apply_commands(
        (
            command(0, 20, prompt_len=1),
            command(1, 21, prompt_len=1),
            command(2, 22, prompt_len=1),
            command(3, 23, prompt_len=1),
            command(4, 24, prompt_len=1),
        ),
        now_ns=0,
    )

    first_plan, _ = _commit_fake_step(scheduler, 0)
    assert tuple(
        item.request_id for item in first_plan.items
    ) == (20, 21, 22, 23)

    scheduler.apply_commands((), now_ns=3)
    txn = scheduler.begin_step(now_ns=3)
    assert txn is not None

    assert tuple(
        item.request_id for item in txn.plan.items
    ) == (20, 21, 22, 24)
    assert tuple(
        item.work_class for item in txn.plan.items
    ) == (
        WorkClass.DECODE,
        WorkClass.DECODE,
        WorkClass.DECODE,
        WorkClass.PREFILL,
    )
    assert len(
        {
            item.request_id
            for item in txn.plan.items
        }
    ) == len(txn.plan.items)

    scheduler.rollback(txn)


def test_begin_step_reserves_kv_without_publishing_table() -> None:
    scheduler = make_scheduler(
        active=1,
        rows=1,
        tokens=4,
        chunk=4,
        model_len=8,
        kv_page_size=4,
        kv_num_pages=2,
    )
    scheduler.apply_commands(
        (command(0, 10, prompt_len=3),),
        now_ns=0,
    )

    initial_free = scheduler.kv_manager.allocator.free_page_count
    txn = scheduler.begin_step(now_ns=0)

    assert txn is not None
    assert txn.kv.state is KvTxnState.PREPARED
    assert scheduler.kv_manager.page_table(10) == ()
    assert txn.kv.staged_page_table(10)
    assert (
        scheduler.kv_manager.allocator.free_page_count
        == initial_free - 1
    )

    scheduler.rollback(txn)

    assert txn.kv.state is KvTxnState.ROLLED_BACK
    assert scheduler.kv_manager.page_table(10) == ()
    assert (
        scheduler.kv_manager.allocator.free_page_count
        == initial_free
    )


def test_kv_capacity_failure_is_atomic_and_returns_no_schedule() -> None:
    scheduler = make_scheduler(
        active=2,
        rows=2,
        tokens=10,
        chunk=8,
        model_len=8,
        kv_page_size=4,
        kv_num_pages=1,
    )
    scheduler.apply_commands(
        (
            command(0, 10, prompt_len=5),
            command(1, 11, prompt_len=5),
        ),
        now_ns=0,
    )

    before = committed_state(scheduler)
    before_allocator = scheduler.kv_manager.allocator.snapshot()

    assert scheduler.begin_step(now_ns=0) is None

    assert committed_state(scheduler) == before
    assert (
        scheduler.kv_manager.allocator.snapshot()
        == before_allocator
    )
    assert scheduler.kv_manager.page_table(10) == ()
    assert scheduler.kv_manager.page_table(11) == ()
    assert not scheduler.has_open_transaction


def test_batch_prepare_failure_can_roll_back_the_combined_plan() -> None:
    scheduler = make_scheduler(
        active=1,
        model_len=8,
        kv_page_size=4,
        kv_num_pages=2,
    )
    scheduler.apply_commands(
        (command(0, 10, prompt_len=5),),
        now_ns=0,
    )

    before_allocator = scheduler.kv_manager.allocator.snapshot()
    txn = scheduler.begin_step(now_ns=0)

    assert txn is not None

    with pytest.raises(ValueError, match="max_model_len"):
        BatchBuilder(max_model_len=2).prepare(
            txn.plan,
            (scheduler.get_request(10),),
            kv=txn.kv,
        )

    scheduler.rollback(txn)

    assert txn.kv.state is KvTxnState.ROLLED_BACK
    assert scheduler.kv_manager.page_table(10) == ()

    after_allocator = scheduler.kv_manager.allocator.snapshot()
    assert after_allocator[0] == before_allocator[0]
    assert after_allocator[2] == before_allocator[2]
    assert after_allocator[1][0] == (
        before_allocator[1][0] + 1
    )


def test_kv_fallback_runs_decode_that_reuses_a_partial_page() -> None:
    scheduler = make_scheduler(
        active=2,
        rows=2,
        tokens=8,
        chunk=8,
        model_len=8,
        kv_page_size=4,
        kv_num_pages=2,
    )
    scheduler.apply_commands(
        (
            command(0, 10, prompt_len=4, max_new_tokens=2),
            command(1, 11, prompt_len=3, max_new_tokens=2),
        ),
        now_ns=0,
    )

    _commit_fake_step(scheduler, 0)

    scheduler.apply_commands((), now_ns=3)
    txn = scheduler.begin_step(now_ns=3)

    assert txn is not None
    assert txn.kv_fallback
    assert tuple(
        item.request_id for item in txn.plan.items
    ) == (11,)
    assert txn.plan.items[0].work_class is WorkClass.DECODE

    batch = BatchBuilder(max_model_len=8).prepare(
        txn.plan,
        (scheduler.get_request(11),),
        kv=txn.kv,
    )
    scheduler.mark_dispatched(txn, now_ns=4)
    scheduler.commit_step(
        txn,
        CpuFakeExecutor(token_id=99).execute(batch),
        now_ns=5,
    )

    assert scheduler.get_request(11).status is RequestStatus.FINISHED
    assert scheduler.kv_manager.allocator.free_page_count == 1

    scheduler.apply_commands((), now_ns=6)
    next_txn = scheduler.begin_step(now_ns=6)

    assert next_txn is not None
    assert not next_txn.kv_fallback
    assert tuple(
        item.request_id for item in next_txn.plan.items
    ) == (10,)

    scheduler.rollback(next_txn)


def test_kv_fallback_compacts_a_later_waiting_prefill_slot() -> None:
    scheduler = make_scheduler(
        active=2,
        rows=2,
        tokens=8,
        chunk=8,
        model_len=8,
        kv_page_size=4,
        kv_num_pages=1,
    )
    scheduler.apply_commands(
        (
            command(0, 10, prompt_len=5),
            command(1, 11, prompt_len=1),
        ),
        now_ns=0,
    )

    txn = scheduler.begin_step(now_ns=0)

    assert txn is not None
    assert txn.kv_fallback
    assert tuple(
        item.request_id for item in txn.plan.items
    ) == (11,)
    assert txn.plan.items[0].request_slot == 0

    scheduler.rollback(txn)
    assert scheduler.waiting_request_ids == (10, 11)

    txn = scheduler.begin_step(now_ns=0)
    assert txn is not None
    assert txn.kv_fallback

    batch = BatchBuilder(max_model_len=8).prepare(
        txn.plan,
        (scheduler.get_request(11),),
        kv=txn.kv,
    )
    scheduler.mark_dispatched(txn, now_ns=1)

    assert scheduler.waiting_request_ids == (10,)
    assert scheduler.get_request(11).active_slot == 0
    assert scheduler.get_request(10).active_slot is None

    scheduler.commit_step(
        txn,
        CpuFakeExecutor(token_id=99).execute(batch),
        now_ns=2,
    )


def test_scheduler_kv_plan_can_be_lowered_into_prepared_batch() -> None:
    scheduler = make_scheduler(
        active=1,
        rows=1,
        tokens=4,
        chunk=4,
        model_len=8,
        kv_page_size=4,
        kv_num_pages=2,
    )

    scheduler.apply_commands(
        (command(0, 10, prompt_len=3),),
        now_ns=0,
    )

    txn = scheduler.begin_step(now_ns=0)
    assert txn is not None

    batch = BatchBuilder(max_model_len=8).prepare(
        txn.plan,
        (scheduler.get_request(10),),
        kv=txn.kv,
    )

    assert batch.has_kv_metadata
    assert batch.kv_page_table_ids == (0,)
    assert batch.kv_page_table_offsets == (0, 1)
    assert batch.kv_slot_mapping == (0, 1, 2)

    scheduler.rollback(txn)