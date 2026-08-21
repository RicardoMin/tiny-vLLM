from __future__ import annotations

import random

import pytest

from tinyvllm.cache import (
    KvAppend,
    KvPageManager,
    KvPlanTxn,
    KvTxnState,
    PageAllocator,
    PageRef,
    PageState,
)


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



def _commit(
    manager: KvPageManager,
    plan_id: int,
    *appends: KvAppend,
) -> tuple[KvPlanTxn, tuple[PageRef, ...]]:
    txn = manager.begin_plan(
        plan_id=plan_id,
        appends=tuple(appends),
    )
    assert txn is not None

    txn.commit_before_dispatch()
    written_refs = txn.complete_writes()
    return txn, written_refs


def test_kv_plan_stages_pages_and_linear_slots() -> None:
    manager = KvPageManager(page_size=4, num_pages=4)

    txn = manager.begin_plan(
        plan_id=1,
        appends=(KvAppend(10, 0, 6),),
    )

    assert txn is not None
    assert txn.state is KvTxnState.PREPARED
    assert tuple(
        ref.page_id for ref in txn.staged_page_table(10)
    ) == (0, 1)
    assert txn.slot_mapping(10) == (0, 1, 2, 3, 4, 5)

    # Staging has not published anything yet.
    assert manager.page_table(10) == ()
    assert manager.valid_token_count(10) == 0
    assert manager.allocator.free_page_count == 2


def test_dispatch_publishes_table_before_write_progress() -> None:
    manager = KvPageManager(page_size=4, num_pages=3)
    txn = manager.begin_plan(
        plan_id=1,
        appends=(KvAppend(10, 0, 6),),
    )
    assert txn is not None

    staged_table = txn.staged_page_table(10)
    txn.commit_before_dispatch()

    assert txn.state is KvTxnState.DISPATCHED
    assert manager.page_table(10) == staged_table
    assert manager.valid_token_count(10) == 0

    assert all(
        manager.allocator.page_state(ref)
        is PageState.PRIVATE_MUTABLE
        for ref in staged_table
    )

    assert txn.complete_writes() == staged_table
    assert txn.state is KvTxnState.COMMITTED
    assert manager.valid_token_count(10) == 6


def test_partial_page_is_reused_and_full_boundary_allocates() -> None:
    manager = KvPageManager(page_size=4, num_pages=3)
    _commit(manager, 1, KvAppend(10, 0, 6))

    reuse = manager.begin_plan(
        plan_id=2,
        appends=(KvAppend(10, 6, 2),),
    )
    assert reuse is not None
    assert len(reuse.staged_page_table(10)) == 2
    assert reuse.slot_mapping(10) == (6, 7)

    reuse.commit_before_dispatch()
    reuse.complete_writes()

    grow = manager.begin_plan(
        plan_id=3,
        appends=(KvAppend(10, 8, 1),),
    )
    assert grow is not None
    assert len(grow.staged_page_table(10)) == 3
    assert grow.slot_mapping(10) == (8,)


def test_multi_request_reservation_is_all_or_none() -> None:
    manager = KvPageManager(page_size=4, num_pages=2)
    before = manager.allocator.snapshot()

    assert manager.begin_plan(
        plan_id=1,
        appends=(
            KvAppend(10, 0, 5),
            KvAppend(11, 0, 1),
        ),
    ) is None

    assert manager.allocator.snapshot() == before
    assert manager.page_table(10) == ()
    assert manager.page_table(11) == ()


def test_invalid_later_append_is_atomic() -> None:
    manager = KvPageManager(page_size=4, num_pages=3)
    before = manager.allocator.snapshot()

    with pytest.raises(ValueError, match="valid KV length"):
        manager.begin_plan(
            plan_id=1,
            appends=(
                KvAppend(10, 0, 5),
                KvAppend(11, 1, 1),
            ),
        )

    assert manager.allocator.snapshot() == before
    assert manager.page_table(10) == ()


def test_rollback_preserves_committed_table_and_progress() -> None:
    manager = KvPageManager(page_size=4, num_pages=3)
    _commit(manager, 1, KvAppend(10, 0, 4))

    before_table = manager.page_table(10)
    before_free_ids, _, before_states = manager.allocator.snapshot()

    txn = manager.begin_plan(
        plan_id=2,
        appends=(KvAppend(10, 4, 1),),
    )
    assert txn is not None
    txn.rollback()

    free_ids, _, states = manager.allocator.snapshot()

    assert txn.state is KvTxnState.ROLLED_BACK
    assert manager.page_table(10) == before_table
    assert manager.valid_token_count(10) == 4
    assert free_ids == before_free_ids
    assert states == before_states


def test_one_reservation_is_partitioned_between_requests() -> None:
    manager = KvPageManager(page_size=4, num_pages=4)
    txn = manager.begin_plan(
        plan_id=1,
        appends=(
            KvAppend(10, 0, 5),
            KvAppend(11, 0, 3),
        ),
    )
    assert txn is not None

    assert tuple(
        ref.page_id for ref in txn.staged_page_table(10)
    ) == (0, 1)
    assert tuple(
        ref.page_id for ref in txn.staged_page_table(11)
    ) == (2,)
    assert txn.slot_mapping(10) == (0, 1, 2, 3, 4)
    assert txn.slot_mapping(11) == (8, 9, 10)


def test_completed_write_returns_only_pages_touched_this_step() -> None:
    manager = KvPageManager(page_size=4, num_pages=4)
    _commit(manager, 1, KvAppend(10, 0, 6))
    _, written_refs = _commit(manager, 2, KvAppend(10, 6, 3))

    assert tuple(
        ref.page_id for ref in written_refs
    ) == (1, 2)


def test_release_request_returns_all_pages() -> None:
    manager = KvPageManager(page_size=4, num_pages=3)
    _commit(manager, 1, KvAppend(10, 0, 5))

    manager.release_request(10)

    assert manager.page_table(10) == ()
    assert manager.valid_token_count(10) == 0
    assert manager.allocator.free_page_count == 3
    assert manager.allocator.snapshot()[0] == (2, 1, 0)


def test_kv_transaction_lifecycle_is_single_use() -> None:
    manager = KvPageManager(page_size=4, num_pages=2)

    with pytest.raises(ValueError, match="twice"):
        manager.begin_plan(
            plan_id=1,
            appends=(
                KvAppend(10, 0, 1),
                KvAppend(10, 0, 1),
            ),
        )

    txn = manager.begin_plan(
        plan_id=2,
        appends=(KvAppend(10, 0, 1),),
    )
    assert txn is not None

    with pytest.raises(RuntimeError, match="already open"):
        manager.begin_plan(
            plan_id=3,
            appends=(KvAppend(11, 0, 1),),
        )

    with pytest.raises(ValueError, match="expected state"):
        txn.complete_writes()

    txn.commit_before_dispatch()

    with pytest.raises(ValueError, match="expected state"):
        txn.rollback()

    with pytest.raises(ValueError, match="expected state"):
        txn.commit_before_dispatch()

    txn.complete_writes()

    with pytest.raises(ValueError, match="expected state"):
        txn.complete_writes()


@pytest.mark.parametrize(
    "args",
    (
        (True, 0, 1),
        (1, -1, 1),
        (1, 0, 0),
    ),
)
def test_kv_append_rejects_invalid_fields(
    args: tuple[int, int, int],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        KvAppend(*args)


def test_required_new_pages_is_a_pure_capacity_query() -> None:
    manager = KvPageManager(page_size=4, num_pages=2)
    before = manager.allocator.snapshot()

    assert manager.free_page_count == 2
    assert manager.required_new_pages(
        KvAppend(10, 0, 3)
    ) == 1
    assert manager.allocator.snapshot() == before

    _commit(manager, 1, KvAppend(10, 0, 3))

    assert manager.required_new_pages(
        KvAppend(10, 3, 1)
    ) == 0
    assert manager.free_page_count == 1