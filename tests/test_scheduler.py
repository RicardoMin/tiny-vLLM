from __future__ import annotations

import pytest
import copy

from tinyvllm.request import RequestStatus, SamplingParams
from tinyvllm.scheduler import AddCommand, Scheduler, TxnState


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
) -> Scheduler:
    return Scheduler(
        max_active_requests=active,
        max_step_requests=active if rows is None else rows,
        max_scheduled_tokens=tokens,
        prefill_chunk_size=chunk,
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