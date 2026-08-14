from __future__ import annotations

import pytest

from tinyvllm.request import RequestStatus, SamplingParams
from tinyvllm.scheduler import AddCommand, Scheduler


def command(
    seq: int,
    request_id: int,
    *,
    arrival_ns: int = 0,
    max_new_tokens: int = 4,
) -> AddCommand:
    return AddCommand(
        seq=seq,
        request_id=request_id,
        prompt_token_ids=(1, 2, 3),
        sampling=SamplingParams(
            max_new_tokens=max_new_tokens
        ),
        arrival_ns=arrival_ns,
    )


def test_command_cutoff_is_external() -> None:
    scheduler = Scheduler()

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
    scheduler = Scheduler()

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
    scheduler = Scheduler()

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
    scheduler = Scheduler()

    scheduler.apply_commands(
        (
            command(0, 10, arrival_ns=100),
            command(1, 11, arrival_ns=10),
        ),
        now_ns=10,
    )

    assert scheduler.waiting_request_ids == (11,)


def test_new_arrivals_use_fifo_order() -> None:
    scheduler = Scheduler()

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
    scheduler = Scheduler()

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


def test_clock_cannot_move_backwards() -> None:
    scheduler = Scheduler()

    scheduler.apply_commands(
        (command(0, 10),),
        now_ns=10,
    )

    with pytest.raises(ValueError, match="backwards"):
        scheduler.apply_commands((), now_ns=9)