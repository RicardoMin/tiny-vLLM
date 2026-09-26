"""Core contracts of the CPU block pool and transactional request tables."""

import pytest

from tinyvllm.cache import (
    BlockAppend,
    BlockManager,
    BlockPool,
    BlockReservation,
    BlockState,
    BlockTxnState,
)


def _commit(manager, plan_id, *appends):
    txn = manager.begin_plan(plan_id=plan_id, appends=appends)
    assert txn is not None
    txn.commit_before_dispatch()
    txn.complete_writes()
    return txn


def _assert_metadata_closed(txn, request_id):
    for query in (txn.staged_block_table, txn.slot_mapping):
        with pytest.raises(RuntimeError, match="no longer readable"):
            query(request_id)


def test_pool_capacity_failure_and_zero_count_are_atomic():
    pool = BlockPool(num_blocks=2)
    before = pool.snapshot()
    assert pool.reserve(3) is None
    with pytest.raises(ValueError):
        pool.reserve(0)
    assert pool.snapshot() == before
    assert pool.open_reservation_count == 0


@pytest.mark.parametrize("recycle", ["rollback", "release"])
def test_recycling_invalidates_old_generation(recycle):
    pool = BlockPool(num_blocks=1)
    reservation = pool.reserve(1)
    (old,) = reservation.block_refs
    assert pool.state(old) is BlockState.RESERVED
    if recycle == "rollback":
        pool.rollback(reservation)
    else:
        with pytest.raises(RuntimeError, match="not active"):
            pool.release(old)
        pool.commit(reservation)
        assert pool.state(old) is BlockState.ACTIVE
        pool.release(old)
        with pytest.raises(RuntimeError, match="not active"):
            pool.release(old)
    assert pool.free_block_count == 1
    fresh = pool.reserve(1).block_refs[0]
    assert fresh.block_id == old.block_id
    assert fresh.generation > old.generation
    with pytest.raises(RuntimeError, match="stale"):
        pool.state(old)
    with pytest.raises(RuntimeError, match="stale"):
        pool.release(old)
    assert pool.state(fresh) is BlockState.RESERVED


def test_reservation_identity_single_open_and_single_use():
    pool = BlockPool(num_blocks=2)
    reservation = pool.reserve(1)
    foreign = BlockPool(num_blocks=2).reserve(1)
    forged = BlockReservation(reservation.block_refs)
    before = pool.snapshot()
    with pytest.raises(RuntimeError, match="already open"):
        pool.reserve(1)
    for invalid in (foreign, forged):
        for operation in (pool.commit, pool.rollback):
            with pytest.raises(ValueError):
                operation(invalid)
    assert pool.snapshot() == before
    pool.commit(reservation)
    for operation in (pool.commit, pool.rollback):
        with pytest.raises(ValueError):
            operation(reservation)
    assert pool.open_reservation_count == 0


def test_mixed_append_isolated_tables_slots_and_write_progress():
    manager = BlockManager(block_size=4, num_blocks=5)
    _commit(manager, 1, BlockAppend(10, 0, 6))
    old_table = manager.block_table(10)
    txn = manager.begin_plan(
        plan_id=2, appends=(BlockAppend(10, 6, 3), BlockAppend(11, 0, 2))
    )
    first, second = txn.staged_block_table(10), txn.staged_block_table(11)
    assert first[:2] == old_table
    assert len(first) == 3 and len(second) == 1
    assert {ref.block_id for ref in first}.isdisjoint(
        ref.block_id for ref in second
    )
    assert txn.slot_mapping(10) == (
        first[1].block_id * 4 + 2, first[1].block_id * 4 + 3,
        first[2].block_id * 4,
    )
    assert txn.slot_mapping(11) == (
        second[0].block_id * 4, second[0].block_id * 4 + 1,
    )
    assert manager.block_table(10) == old_table
    assert manager.block_table(11) == ()
    txn.commit_before_dispatch()
    assert manager.block_table(10) == first
    assert manager.block_table(11) == second
    assert (manager.valid_token_count(10), manager.valid_token_count(11)) == (6, 0)
    txn.complete_writes()
    assert (manager.valid_token_count(10), manager.valid_token_count(11)) == (9, 2)
    _assert_metadata_closed(txn, 10)
    with pytest.raises(ValueError):
        txn.complete_writes()


def test_zero_new_block_batch_works_with_exhausted_pool():
    manager = BlockManager(block_size=4, num_blocks=2)
    _commit(manager, 1, BlockAppend(10, 0, 2), BlockAppend(11, 0, 3))
    assert manager.free_block_count == 0
    txn = manager.begin_plan(
        plan_id=2, appends=(BlockAppend(10, 2, 2), BlockAppend(11, 3, 1))
    )
    assert txn is not None
    assert manager.pool.open_reservation_count == 0
    txn.commit_before_dispatch()
    txn.complete_writes()
    assert (manager.valid_token_count(10), manager.valid_token_count(11)) == (4, 4)


def test_multi_request_capacity_failure_is_atomic():
    manager = BlockManager(block_size=4, num_blocks=2)
    before = manager.pool.snapshot()
    assert manager.begin_plan(
        plan_id=1, appends=(BlockAppend(10, 0, 5), BlockAppend(11, 0, 1))
    ) is None
    assert manager.pool.snapshot() == before
    assert manager.block_table(10) == manager.block_table(11) == ()
    _commit(manager, 2, BlockAppend(11, 0, 1))


@pytest.mark.parametrize("invalid", [BlockAppend(11, 1, 1), BlockAppend(10, 0, 1), object()])
def test_invalid_later_append_does_not_reserve_or_publish(invalid):
    manager = BlockManager(block_size=4, num_blocks=3)
    before = manager.pool.snapshot()
    with pytest.raises((TypeError, ValueError)):
        manager.begin_plan(plan_id=1, appends=(BlockAppend(10, 0, 5), invalid))
    assert manager.pool.snapshot() == before
    assert manager.block_table(10) == ()
    _commit(manager, 2, BlockAppend(10, 0, 1))


def test_pre_dispatch_rollback_preserves_committed_state():
    manager = BlockManager(block_size=4, num_blocks=3)
    _commit(manager, 1, BlockAppend(10, 0, 3))
    before = manager.block_table(10)
    txn = manager.begin_plan(plan_id=2, appends=(BlockAppend(10, 3, 3),))
    with pytest.raises(RuntimeError, match="already open"):
        manager.begin_plan(plan_id=3, appends=(BlockAppend(11, 0, 1),))
    with pytest.raises(RuntimeError):
        manager.release_request(10)
    txn.rollback()
    assert txn.state is BlockTxnState.ROLLED_BACK
    _assert_metadata_closed(txn, 10)
    assert manager.block_table(10) == before
    assert manager.valid_token_count(10) == 3
    assert manager.free_block_count == 2
    with pytest.raises(ValueError):
        txn.rollback()


@pytest.mark.parametrize("dispatched", [False, True])
def test_teardown_closes_open_transaction_and_releases_all(dispatched):
    manager = BlockManager(block_size=4, num_blocks=3)
    _commit(manager, 1, BlockAppend(10, 0, 3))
    txn = manager.begin_plan(plan_id=2, appends=(BlockAppend(11, 0, 5),))
    if dispatched:
        txn.commit_before_dispatch()
        with pytest.raises(ValueError):
            txn.rollback()
    manager.teardown()
    assert txn.state is BlockTxnState.FAILED
    _assert_metadata_closed(txn, 11)
    assert manager.pool.open_reservation_count == 0
    assert manager.free_block_count == 3
    assert manager.block_table(10) == manager.block_table(11) == ()
    assert manager.valid_token_count(10) == manager.valid_token_count(11) == 0


def test_latest_completed_transaction_can_fail_after_capacity_rejection():
    manager = BlockManager(block_size=4, num_blocks=1)
    latest = _commit(manager, 1, BlockAppend(10, 0, 4))
    assert manager.begin_plan(plan_id=2, appends=(BlockAppend(11, 0, 1),)) is None
    latest.fail()
    assert latest.state is BlockTxnState.FAILED
    assert manager.free_block_count == 1
    assert manager.block_table(10) == ()


@pytest.mark.parametrize("finish_new", ["open", "complete", "rollback"])
def test_old_and_foreign_transactions_cannot_teardown_current_state(finish_new):
    manager = BlockManager(block_size=4, num_blocks=3)
    old = _commit(manager, 1, BlockAppend(10, 0, 1))
    newer = manager.begin_plan(plan_id=2, appends=(BlockAppend(11, 0, 1),))
    if finish_new == "complete":
        newer.commit_before_dispatch()
        newer.complete_writes()
    elif finish_new == "rollback":
        newer.rollback()
    other = BlockManager(block_size=4, num_blocks=1)
    foreign = _commit(other, 1, BlockAppend(12, 0, 1))
    before = manager.pool.snapshot(), manager.block_table(10), manager.block_table(11)
    for invalid in (old, foreign):
        with pytest.raises(ValueError):
            manager.teardown(invalid)
        assert invalid.state is BlockTxnState.COMMITTED
    with pytest.raises(ValueError):
        old.fail()
    assert (manager.pool.snapshot(), manager.block_table(10), manager.block_table(11)) == before
    assert other.valid_token_count(12) == 1


def test_release_request_returns_only_its_blocks():
    manager = BlockManager(block_size=4, num_blocks=3)
    _commit(manager, 1, BlockAppend(10, 0, 5), BlockAppend(11, 0, 1))
    other_table = manager.block_table(11)
    manager.release_request(10)
    assert manager.block_table(10) == ()
    assert manager.valid_token_count(10) == 0
    assert manager.free_block_count == 2
    assert manager.block_table(11) == other_table
    manager.release_request(11)
    assert manager.free_block_count == 3
