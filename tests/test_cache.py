from __future__ import annotations

import random

import pytest

from tinyvllm.cache import PageAllocator, PageRef, PageState


def test_allocator_starts_with_deterministic_free_stack() -> None:
    allocator = PageAllocator(num_pages=3)

    assert allocator.num_pages == 3
    assert allocator.free_page_count == 3
    assert allocator.open_reservation_count == 0
    assert allocator.snapshot()[0] == (2, 1, 0)

    assert all(
        allocator.page_state(PageRef(page_id, 0)) is PageState.FREE
        for page_id in range(3)
    )


def test_reserve_commit_and_release_follow_page_ownership() -> None:
    allocator = PageAllocator(num_pages=3)
    reservation = allocator.reserve(count=2, txn_id=7)

    assert reservation is not None
    assert tuple(
        ref.page_id for ref in reservation.page_refs
    ) == (0, 1)

    assert all(
        allocator.page_state(ref) is PageState.RESERVED
        for ref in reservation.page_refs
    )

    assert allocator.free_page_count == 1
    assert allocator.open_reservation_count == 1

    allocator.commit(reservation)

    assert allocator.open_reservation_count == 0

    assert all(
        allocator.page_state(ref) is PageState.PRIVATE_MUTABLE
        for ref in reservation.page_refs
    )

    allocator.release(reservation.page_refs[0])

    assert allocator.free_page_count == 2
    assert (
        allocator.page_state(reservation.page_refs[0])
        is PageState.FREE
    )


def test_reserve_is_all_or_nothing_when_capacity_is_insufficient() -> None:
    allocator = PageAllocator(num_pages=2)
    before = allocator.snapshot()

    assert allocator.reserve(count=3, txn_id=1) is None

    assert allocator.snapshot() == before
    assert allocator.open_reservation_count == 0


def test_rollback_restores_free_stack_order_without_rewinding_generation() -> None:
    allocator = PageAllocator(num_pages=4)
    before = allocator.snapshot()

    reservation = allocator.reserve(count=2, txn_id=9)
    assert reservation is not None

    allocator.rollback(reservation)

    free_ids, generations, states = allocator.snapshot()

    assert free_ids == before[0]
    assert generations[0] == before[1][0] + 1
    assert generations[1] == before[1][1] + 1
    assert states == before[2]
    assert allocator.open_reservation_count == 0


def test_generation_rejects_stale_ref_after_release_and_reuse() -> None:
    allocator = PageAllocator(num_pages=1)

    first = allocator.reserve(count=1, txn_id=1)
    assert first is not None

    allocator.commit(first)

    old_ref = first.page_refs[0]
    allocator.release(old_ref)

    second = allocator.reserve(count=1, txn_id=2)
    assert second is not None

    new_ref = second.page_refs[0]

    assert new_ref.page_id == old_ref.page_id
    assert new_ref.generation == old_ref.generation + 1

    with pytest.raises(RuntimeError, match="stale PageRef"):
        allocator.page_state(old_ref)

    with pytest.raises(RuntimeError, match="stale PageRef"):
        allocator.release(old_ref)


def test_reservation_lifecycle_is_single_use() -> None:
    allocator = PageAllocator(num_pages=2)

    reservation = allocator.reserve(count=1, txn_id=4)
    assert reservation is not None

    with pytest.raises(ValueError, match="transaction already owns"):
        allocator.reserve(count=1, txn_id=4)

    allocator.commit(reservation)

    with pytest.raises(ValueError, match="not open"):
        allocator.commit(reservation)

    with pytest.raises(ValueError, match="not open"):
        allocator.rollback(reservation)


def test_reserved_page_cannot_be_released_directly() -> None:
    allocator = PageAllocator(num_pages=1)

    reservation = allocator.reserve(count=1, txn_id=3)
    assert reservation is not None

    with pytest.raises(RuntimeError, match="private mutable"):
        allocator.release(reservation.page_refs[0])


def test_release_does_not_duplicate_page_ids() -> None:
    allocator = PageAllocator(num_pages=2)

    reservation = allocator.reserve(count=1, txn_id=5)
    assert reservation is not None

    allocator.commit(reservation)

    ref = reservation.page_refs[0]
    allocator.release(ref)

    with pytest.raises(RuntimeError, match="private mutable"):
        allocator.release(ref)

    next_reservation = allocator.reserve(count=2, txn_id=6)
    assert next_reservation is not None

    assert len({
        ref.page_id for ref in next_reservation.page_refs
    }) == 2


def test_random_lifecycle_preserves_single_page_ownership() -> None:
    rng = random.Random(20260818)
    allocator = PageAllocator(num_pages=8)

    open_reservations = {}
    private_refs = {}
    next_txn_id = 1
    last_generation = [0] * allocator.num_pages

    for _ in range(500):
        action = rng.randrange(4)

        if action == 0 or (
            not open_reservations and not private_refs
        ):
            count = rng.randint(1, 3)
            txn_id = next_txn_id
            next_txn_id += 1

            reservation = allocator.reserve(
                count=count,
                txn_id=txn_id,
            )

            if reservation is not None:
                open_reservations[txn_id] = reservation

                for ref in reservation.page_refs:
                    assert (
                        ref.generation
                        > last_generation[ref.page_id]
                    )
                    last_generation[ref.page_id] = ref.generation

        elif action == 1 and open_reservations:
            txn_id = rng.choice(tuple(open_reservations))
            reservation = open_reservations.pop(txn_id)

            allocator.commit(reservation)

            private_refs.update(
                (ref.page_id, ref)
                for ref in reservation.page_refs
            )

        elif action == 2 and open_reservations:
            txn_id = rng.choice(tuple(open_reservations))
            reservation = open_reservations.pop(txn_id)
            allocator.rollback(reservation)

        elif private_refs:
            page_id = rng.choice(tuple(private_refs))
            ref = private_refs.pop(page_id)
            allocator.release(ref)

        free_ids, _, states = allocator.snapshot()

        assert len(free_ids) == len(set(free_ids))

        assert set(free_ids) == {
            page_id
            for page_id, state in enumerate(states)
            if state is PageState.FREE
        }

        reserved_ids = {
            ref.page_id
            for reservation in open_reservations.values()
            for ref in reservation.page_refs
        }

        assert reserved_ids.isdisjoint(private_refs)

        assert reserved_ids == {
            page_id
            for page_id, state in enumerate(states)
            if state is PageState.RESERVED
        }

        assert set(private_refs) == {
            page_id
            for page_id, state in enumerate(states)
            if state is PageState.PRIVATE_MUTABLE
        }

    for reservation in tuple(open_reservations.values()):
        allocator.rollback(reservation)

    for ref in tuple(private_refs.values()):
        allocator.release(ref)

    assert allocator.free_page_count == allocator.num_pages
    assert allocator.open_reservation_count == 0


@pytest.mark.parametrize(
    ("count", "txn_id"),
    [
        (True, 1),
        (1, True),
        (-1, 1),
        (1, -1),
    ],
)
def test_reserve_rejects_invalid_integer_arguments(
    count: int,
    txn_id: int,
) -> None:
    allocator = PageAllocator(num_pages=2)

    with pytest.raises((TypeError, ValueError)):
        allocator.reserve(
            count=count,
            txn_id=txn_id,
        )