from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from collections.abc import Iterable

import pytest

from tinyvllm.request import (
    RequestStatus,
    SamplingParams,
    WorkClass
)
from tinyvllm.scheduler import (
    AddCommand,
    CancelCommand,
    Scheduler,
    TxnState,
)
from tinyvllm.batch import (
    BatchBuilder,
    StepResult,
    StepMode,
)
from engine_helpers import CpuFakeExecutor
from tinyvllm.cache import BlockTxnState


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
    kv_block_size: int = 16,
    kv_num_blocks: int | None = None,
    eos_token_ids: Iterable[int] | None = None,
) -> Scheduler:
    return Scheduler(
        max_active_requests=active,
        max_step_requests=active if rows is None else rows,
        max_scheduled_tokens=tokens,
        prefill_chunk_size=chunk,
        max_model_len=model_len,
        prefill_reserve_tokens=reserve_tokens,
        prefill_reserve_requests=reserve_rows,
        kv_block_size=kv_block_size,
        kv_num_blocks=kv_num_blocks,
        eos_token_ids=eos_token_ids,
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


def test_scheduler_requires_matching_prefill_reservations() -> None:
    with pytest.raises(ValueError, match="both be zero or positive"):
        make_scheduler(reserve_tokens=1, reserve_rows=0)


def test_prepare_is_pure_and_uses_lowest_slots() -> None:
    scheduler = make_scheduler(active=2)
    scheduler.apply_commands(
        (command(0, 10), command(1, 11), command(2, 12)),
        now_ns=0,
    )
    before = committed_state(scheduler)
    initial_free = scheduler.block_manager.free_block_count

    txn = scheduler.begin_step(now_ns=0)

    assert txn is not None
    assert [
        (item.request_id, item.request_slot)
        for item in txn.plan.items
    ] == [(10, 0), (11, 1)]
    assert committed_state(scheduler) == before

    assert txn.kv.state is BlockTxnState.PREPARED
    assert scheduler.block_manager.block_table(10) == ()
    assert txn.kv.staged_block_table(10)
    assert scheduler.block_manager.free_block_count < initial_free
    scheduler.rollback(txn)
    assert committed_state(scheduler) == before
    assert scheduler.block_manager.free_block_count == initial_free
    retry = scheduler.begin_step(now_ns=0)
    assert retry.plan.plan_id > txn.plan.plan_id
    scheduler.rollback(retry)


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
    assert txn.kv.state is BlockTxnState.DISPATCHED
    assert scheduler.block_manager.block_table(10)
    assert scheduler.block_manager.valid_token_count(10) == 0
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

    assert scheduler.request_ids == ()
    assert scheduler.completed_request_ids == ()
    assert scheduler.waiting_request_ids == ()
    assert txn.state is TxnState.FAILED
    assert txn.kv.state is BlockTxnState.FAILED
    assert scheduler.block_manager.valid_token_count(10) == 0
    assert not scheduler.has_open_transaction


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

    # Terminal publication returns KV before the caller consumes the result.
    manager = scheduler.block_manager
    assert manager.block_table(10) == ()
    assert manager.valid_token_count(10) == 0
    assert manager.free_block_count == manager.pool.num_blocks
    scheduler.apply_commands((), now_ns=15)
    assert scheduler.begin_step(now_ns=15) is None
    event, = scheduler.consume_finished()
    assert event.output_token_ids == (99, 99, 99)
    assert scheduler.request_ids == ()


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


def test_batch_prepare_failure_can_roll_back_the_combined_plan() -> None:
    scheduler = make_scheduler(
        active=1,
        model_len=8,
        kv_block_size=4,
        kv_num_blocks=2,
    )
    scheduler.apply_commands(
        (command(0, 10, prompt_len=5),),
        now_ns=0,
    )

    before_pool = scheduler.block_manager.pool.snapshot()
    txn = scheduler.begin_step(now_ns=0)

    assert txn is not None

    with pytest.raises(ValueError, match="max_model_len"):
        BatchBuilder(max_model_len=2).prepare(
            txn.plan,
            (scheduler.get_request(10),),
            kv=txn.kv,
        )

    scheduler.rollback(txn)

    assert txn.kv.state is BlockTxnState.ROLLED_BACK
    assert scheduler.block_manager.block_table(10) == ()

    after_pool = scheduler.block_manager.pool.snapshot()
    assert after_pool[0] == before_pool[0]
    assert after_pool[2] == before_pool[2]
    # The reservation is freed (ids and states restored) but the
    # reserved block's generation is never rewound.
    assert sum(after_pool[1]) == (
        sum(before_pool[1]) + 1
    )


def test_consume_finished_is_fifo_and_bounded() -> None:
    scheduler = make_scheduler(active=2)

    scheduler.apply_commands(
        (
            command(
                0,
                10,
                max_new_tokens=0,
            ),
            command(
                1,
                11,
                max_new_tokens=0,
            ),
        ),
        now_ns=0,
    )

    assert scheduler.completed_request_ids == (10, 11)
    assert scheduler.consume_finished(max_items=0) == ()

    first = scheduler.consume_finished(max_items=1)

    assert tuple(
        event.request_id
        for event in first
    ) == (10,)
    assert scheduler.request_ids == (11,)

    with pytest.raises(FrozenInstanceError):
        first[0].request_id = 99

    second = scheduler.consume_finished()

    assert tuple(
        event.request_id
        for event in second
    ) == (11,)
    assert scheduler.request_ids == ()
    assert scheduler.consume_finished() == ()

    with pytest.raises(ValueError):
        scheduler.consume_finished(max_items=-1)

    with pytest.raises(TypeError):
        scheduler.consume_finished(max_items=True)


def test_individually_impossible_request_is_rejected_atomically() -> None:
    scheduler = make_scheduler(
        active=2,
        model_len=8,
        kv_block_size=4,
        kv_num_blocks=1,
    )

    with pytest.raises(ValueError, match="request exceeds KV capacity"):
        scheduler.apply_commands(
            (
                command(0, 10, prompt_len=1, max_new_tokens=1),
                command(1, 11, prompt_len=4, max_new_tokens=2),
            ),
            now_ns=0,
        )

    assert scheduler.request_ids == ()
    assert scheduler.last_command_seq == -1
    assert scheduler.block_manager.free_block_count == 1


@pytest.mark.parametrize(
    ("prompt_len", "max_new_tokens", "model_len", "kv_tokens"),
    (
        (3, 0, 8, 0),
        (3, 1, 8, 3),
        (3, 2, 8, 4),
        (3, 3, 8, 5),
        (3, 99, 8, 7),
        (7, 99, 8, 7),
        (8, 1, 8, 0),
    ),
)
def test_admission_accounts_for_last_token_and_model_limit(
    prompt_len: int,
    max_new_tokens: int,
    model_len: int,
    kv_tokens: int,
) -> None:
    scheduler = make_scheduler(
        active=1,
        model_len=model_len,
        kv_block_size=1,
        kv_num_blocks=max(1, kv_tokens),
    )
    scheduler.apply_commands(
        (command(0, 10, prompt_len=prompt_len, max_new_tokens=max_new_tokens),),
        now_ns=0,
    )

    for now_ns in range(0, 100, 3):
        if scheduler.get_request(10).terminal:
            break
        _commit_fake_step(scheduler, now_ns)
    else:
        pytest.fail("request did not complete within its KV promise")

    assert scheduler.get_request(10).computed_tokens == kv_tokens
    assert scheduler.block_manager.free_block_count == max(1, kv_tokens)


def test_small_pool_serializes_prompts_instead_of_deadlocking() -> None:
    scheduler = make_scheduler(
        active=2,
        rows=1,
        tokens=2,
        chunk=2,
        kv_block_size=2,
        kv_num_blocks=2,
    )
    scheduler.apply_commands(
        (
            command(0, 10, prompt_len=3, max_new_tokens=1),
            command(1, 11, prompt_len=3, max_new_tokens=1),
        ),
        now_ns=0,
    )

    plans = []
    for now_ns in range(0, 12, 3):
        plan, _ = _commit_fake_step(scheduler, now_ns)
        plans.append(tuple(item.request_id for item in plan.items))

    assert plans == [(10,), (10,), (11,), (11,)]
    assert scheduler.completed_request_ids == (10, 11)
    assert scheduler.block_manager.free_block_count == 2


def test_admission_skips_large_waiter_and_preserves_rollback() -> None:
    scheduler = make_scheduler(
        active=2,
        rows=2,
        tokens=2,
        chunk=1,
        kv_block_size=2,
        kv_num_blocks=3,
    )
    scheduler.apply_commands(
        (command(0, 10, prompt_len=2, max_new_tokens=2),),
        now_ns=0,
    )
    _commit_fake_step(scheduler, 0)
    scheduler.apply_commands(
        (
            command(1, 11, prompt_len=3, max_new_tokens=1),
            command(2, 12, prompt_len=1, max_new_tokens=1),
        ),
        now_ns=3,
    )
    before = committed_state(scheduler)
    txn = scheduler.begin_step(now_ns=3)

    assert txn is not None
    assert tuple(item.request_id for item in txn.plan.items) == (10, 12)
    assert txn.plan.items[1].request_slot == 1
    assert scheduler.waiting_request_ids == (11, 12)
    scheduler.rollback(txn)
    assert committed_state(scheduler) == before

    plan, finished = _commit_fake_step(scheduler, 3)
    assert tuple(item.request_id for item in plan.items) == (10, 12)
    assert finished == (12,)
    assert scheduler.waiting_request_ids == (11,)

    for now_ns in range(6, 24, 3):
        if scheduler.get_request(11).terminal:
            break
        _commit_fake_step(scheduler, now_ns)

    assert scheduler.get_request(10).terminal
    assert scheduler.get_request(11).terminal


@pytest.mark.parametrize("peek_before_promotion", (False, True))
def test_cancelled_future_id_can_be_reused_without_early_arrival(
    peek_before_promotion: bool,
) -> None:
    scheduler = make_scheduler()
    scheduler.apply_commands(
        (command(0, 10, arrival_ns=100), CancelCommand(1, 10)),
        now_ns=0,
    )
    scheduler.consume_finished()
    scheduler.apply_commands(
        (command(2, 10, arrival_ns=200),),
        now_ns=0,
    )

    if peek_before_promotion:
        assert scheduler.next_arrival_ns == 200
    scheduler.apply_commands((), now_ns=100)
    assert scheduler.get_request(10).status is RequestStatus.PENDING
    assert scheduler.waiting_request_ids == ()
    assert scheduler.next_arrival_ns == 200
    scheduler.apply_commands((), now_ns=200)
    assert scheduler.waiting_request_ids == (10,)
    assert scheduler.next_arrival_ns is None


def test_cancelled_arrival_heap_is_periodically_compacted() -> None:
    scheduler = make_scheduler()
    commands = []
    for request_id in range(200):
        commands.extend(
            (
                command(2 * request_id, request_id, arrival_ns=1000 + request_id),
                CancelCommand(2 * request_id + 1, request_id),
            )
        )
    commands.append(command(400, 200, arrival_ns=2000))
    scheduler.apply_commands(tuple(commands), now_ns=0)
    scheduler.consume_finished()

    assert len(scheduler._pending_arrivals) == 1
    assert scheduler.next_arrival_ns == 2000
    assert scheduler.request_ids == (200,)


def test_arrival_promotion_uses_time_then_fifo_without_head_blocking():
    scheduler = make_scheduler()
    scheduler.apply_commands(
        (command(0, 10, arrival_ns=100), command(1, 11, arrival_ns=10),
         command(2, 12, arrival_ns=10)),
        now_ns=5,
    )
    assert scheduler.waiting_request_ids == ()
    assert all(scheduler.get_request(rid).status is RequestStatus.PENDING
               for rid in (10, 11, 12))
    scheduler.apply_commands((), now_ns=10)
    assert scheduler.waiting_request_ids == (11, 12)
    assert scheduler.get_request(10).status is RequestStatus.PENDING
    scheduler.apply_commands((), now_ns=100)
    assert scheduler.waiting_request_ids == (11, 12, 10)
