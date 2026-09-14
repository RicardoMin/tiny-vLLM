from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from tinyvllm._validation import (
    _check_non_negative_int,
    _check_positive_int,
)


class BlockState(Enum):
    FREE = auto()
    RESERVED = auto()
    ACTIVE = auto()


@dataclass(frozen=True, slots=True)
class BlockRef:
    """Pool-local physical block identity; generation detects reuse."""
    block_id: int
    generation: int

    def __post_init__(self) -> None:
        _check_non_negative_int("block_id", self.block_id)
        _check_non_negative_int("generation", self.generation)


@dataclass(frozen=True, slots=True)
class BlockReservation:
    block_refs: tuple[BlockRef, ...]


@dataclass(frozen=True, slots=True)
class BlockAppend:
    request_id: int
    token_start: int
    query_len: int

    def __post_init__(self) -> None:
        _check_non_negative_int("request_id", self.request_id)
        _check_non_negative_int("token_start", self.token_start)
        _check_positive_int("query_len", self.query_len)


class BlockPool():
    """CPU block ownership for one serial scheduler, without GPU tensors."""

    def __init__(self, *, num_blocks: int) -> None:
        _check_positive_int("num_blocks", num_blocks)
        self._generations = [0] * num_blocks
        self._states = [BlockState.FREE] * num_blocks
        self._free_ids = list(range(num_blocks - 1, -1, -1))
        self._open_reservation: BlockReservation | None = None


    @property
    def num_blocks(self) -> int:
        return len(self._states)

    @property
    def free_block_count(self) -> int:
        return len(self._free_ids)

    @property
    def open_reservation_count(self) -> int:
        return int(self._open_reservation is not None)


    def snapshot(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[BlockState, ...]]:
        """Diagnostic view; allocation order is not a public contract."""
        return (
            tuple(self._free_ids),
            tuple(self._generations),
            tuple(self._states),
        )


    def reserve(self, count: int) -> BlockReservation | None:
        _check_positive_int("count", count)
        if self._open_reservation is not None:
            raise RuntimeError("a reservation is already open")
        if count > self.free_block_count:
            return None

        refs = []
        for _ in range(count):
            block_id = self._free_ids.pop()
            self._generations[block_id] += 1
            self._states[block_id] = BlockState.RESERVED
            refs.append(BlockRef(block_id, self._generations[block_id]))

        reservation = BlockReservation(tuple(refs))
        self._open_reservation = reservation
        return reservation


    def commit(self, reservation: BlockReservation) -> None:
        self._validate_reservation(reservation)
        for ref in reservation.block_refs:
            self._states[ref.block_id] = BlockState.ACTIVE

        self._open_reservation = None


    def rollback(self, reservation: BlockReservation) -> None:
        self._validate_reservation(reservation)

        for ref in reservation.block_refs:
            self._states[ref.block_id] = BlockState.FREE

        self._free_ids.extend(ref.block_id for ref in reversed(reservation.block_refs))
        # Restore availability, but never rewind generation.
        self._open_reservation = None


    def release(self, ref: BlockRef) -> None:
        if self.state(ref) is not BlockState.ACTIVE:
            raise RuntimeError("block is not active")
        self._states[ref.block_id] = BlockState.FREE
        self._free_ids.append(ref.block_id)


    def state(self, ref: BlockRef) -> BlockState:
        if not isinstance(ref, BlockRef):
            raise TypeError("ref must be BlockRef")
        if ref.block_id >= self.num_blocks:
            raise ValueError("block_id is out of range")
        if self._generations[ref.block_id] != ref.generation:
            raise RuntimeError("stale BlockRef generation")

        return self._states[ref.block_id]


    def _validate_reservation(
        self,
        reservation: BlockReservation,
    ) -> None:
        if self._open_reservation is not reservation:
            raise ValueError(
                "reservation is not open in this pool"
            )
        for ref in reservation.block_refs:
            if self.state(ref) is not BlockState.RESERVED:
                raise RuntimeError("block is not reserved")


class BlockTxnState(Enum):
    PREPARED = auto()
    DISPATCHED = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()
    FAILED = auto()


class BlockPlanTxn:
    """One step's staged block-table changes."""

    __slots__ = (
        "_plan_id",
        "_state",
        "_manager",
        "_reservation",
        "_block_tables",
        "_slot_mappings",
        "_token_ends",
    )

    def __init__(
        self,
        *,
        plan_id: int,
        manager: BlockManager,
        reservation: BlockReservation | None,
        block_tables: dict[int, tuple[BlockRef, ...]],
        slot_mappings: dict[int, tuple[int, ...]],
        token_ends: dict[int, int],
    ) -> None:
        self._plan_id = plan_id
        self._state = BlockTxnState.PREPARED
        self._manager = manager
        self._reservation = reservation
        self._block_tables = block_tables
        self._slot_mappings = slot_mappings
        self._token_ends = token_ends


    @property
    def state(self) -> BlockTxnState:
        return self._state

    @property
    def plan_id(self) -> int:
        return self._plan_id

    def _ensure_metadata_readable(self) -> None:
        if self._state not in (
            BlockTxnState.PREPARED,
            BlockTxnState.DISPATCHED,
        ):
            raise RuntimeError(
                "block transaction metadata is no longer readable"
            )


    def staged_block_table(
        self,
        request_id: int,
    ) -> tuple[BlockRef, ...]:
        self._ensure_metadata_readable()
        return self._block_tables[request_id]

    def slot_mapping(self, request_id: int) -> tuple[int, ...]:
        self._ensure_metadata_readable()
        return self._slot_mappings[request_id]

    def commit_before_dispatch(self) -> None:
        self._manager._commit_before_dispatch(self)

    def complete_writes(self) -> None:
        return self._manager._complete_writes(self)

    def rollback(self) -> None:
        self._manager._rollback(self)

    def fail(self) -> None:
        self._manager._fail(self)


class BlockManager:
    """Owns request block tables and stages append work transactionally."""

    def __init__(
        self,
        *,
        block_size: int,
        num_blocks: int,
    ) -> None:
        _check_positive_int("block_size", block_size)

        self.block_size = block_size
        self.pool = BlockPool(num_blocks=num_blocks)

        self._block_tables: dict[int, tuple[BlockRef, ...]] = {}
        self._valid_token_counts: dict[int, int] = {}
        self._open_txn: BlockPlanTxn | None = None
        self._last_completed_txn: BlockPlanTxn | None = None


    def block_table(
        self,
        request_id: int,
    ) -> tuple[BlockRef, ...]:
        return self._block_tables.get(request_id, ())


    def valid_token_count(self, request_id: int) -> int:
        return self._valid_token_counts.get(request_id, 0)


    @property
    def free_block_count(self) -> int:
        return self.pool.free_block_count


    def required_new_blocks(self, append: BlockAppend) -> int:
        """Validate an append and query its growth without changing ownership."""
        if not isinstance(append, BlockAppend):
            raise TypeError("appends must contain BlockAppend values.")
        if append.token_start != self.valid_token_count(append.request_id):
            raise ValueError("token_start does not match valid KV length")

        table = self.block_table(append.request_id)
        token_end = append.token_start + append.query_len
        count = self._blocks_for_tokens(token_end) - len(table)

        if count < 0:
            raise RuntimeError(
                "block table exceeds valid KV length"
            )

        return count


    def begin_plan(
        self,
        *,
        plan_id: int,
        appends: tuple[BlockAppend, ...],
    ) -> BlockPlanTxn | None:
        """Stage request append intervals as one KV transaction."""
        _check_non_negative_int("plan_id", plan_id)

        if self._open_txn is not None:
            raise RuntimeError("a KV transaction is already open")

        appends = tuple(appends)
        if not appends:
            raise ValueError("KV plan must contain at least one append")

        staged: list[
            tuple[int, int, tuple[BlockRef, ...], int, int]
        ] = []
        seen_request_ids: set[int] = set()
        total_new_blocks = 0

        # Validate the complete batch before reserving any physical block.
        for append in appends:
            new_block_count = self.required_new_blocks(append)

            request_id = append.request_id
            token_start = append.token_start
            query_len = append.query_len

            if request_id in seen_request_ids:
                raise ValueError(
                    "request appears twice in one KV plan"
                )
            seen_request_ids.add(request_id)

            table = self.block_table(request_id)
            token_end = token_start + query_len

            staged.append(
                (
                    request_id,
                    token_start,
                    table,
                    token_end,
                    new_block_count,
                )
            )
            total_new_blocks += new_block_count

        reservation = None
        if total_new_blocks:
            reservation = self.pool.reserve(
                count=total_new_blocks
            )
            if reservation is None:
                return None

        reserved_refs = (
            reservation.block_refs
            if reservation is not None
            else ()
        )

        ref_offset = 0
        block_tables: dict[int, tuple[BlockRef, ...]] = {}
        slot_mappings: dict[int, tuple[int, ...]] = {}
        token_ends: dict[int, int] = {}

        for (
            request_id,
            token_start,
            table,
            token_end,
            new_block_count,
        ) in staged:
            new_refs = reserved_refs[
                ref_offset : ref_offset + new_block_count
            ]
            ref_offset += new_block_count

            staged_table = table + new_refs
            block_tables[request_id] = staged_table
            token_ends[request_id] = token_end

            slot_mappings[request_id] = tuple(
                self._slot_for_position(
                    staged_table,
                    position,
                )
                for position in range(token_start, token_end)
            )

        txn = BlockPlanTxn(
            plan_id=plan_id,
            manager=self,
            reservation=reservation,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            token_ends=token_ends,
        )

        self._open_txn = txn
        self._last_completed_txn = None
        return txn


    def release_request(self, request_id: int) -> None:
        if self._open_txn is not None:
            raise RuntimeError(
                "cannot release blocks during a KV transaction"
            )

        table = self._block_tables.pop(request_id, ())
        self._valid_token_counts.pop(request_id, None)

        for ref in reversed(table):
            self.pool.release(ref)


    def teardown(self, txn: BlockPlanTxn | None = None) -> None:
        """Discard all ownership on fail-stop; never restore partially written KV."""
        current = self._open_txn or self._last_completed_txn
        if txn is not None and (
            txn._manager is not self or txn is not current
        ):
            raise ValueError("transaction is not the manager's current step")

        open_txn = self._open_txn
        if open_txn is not None:
            if open_txn.state is BlockTxnState.PREPARED:
                if open_txn._reservation is not None:
                    self.pool.rollback(open_txn._reservation)
            # Close metadata even when teardown() receives no explicit txn.
            open_txn._state = BlockTxnState.FAILED
            self._open_txn = None

        # Blocks are private to requests: no sharing or duplicate ownership.
        for request_id in tuple(self._block_tables):
            self.release_request(request_id)
        self._last_completed_txn = None
        if txn is not None:
            txn._state = BlockTxnState.FAILED


    def _commit_before_dispatch(self, txn: BlockPlanTxn) -> None:
        self._require_open(txn, BlockTxnState.PREPARED)

        if txn._reservation is not None:
            self.pool.commit(txn._reservation)

        self._block_tables.update(txn._block_tables)
        txn._state = BlockTxnState.DISPATCHED


    def _complete_writes(
        self,
        txn: BlockPlanTxn,
    ) -> None:
        self._require_open(txn, BlockTxnState.DISPATCHED)

        self._valid_token_counts.update(txn._token_ends)
        txn._state = BlockTxnState.COMMITTED
        self._open_txn = None
        self._last_completed_txn = txn


    def _rollback(self, txn: BlockPlanTxn) -> None:
        self._require_open(txn, BlockTxnState.PREPARED)

        if txn._reservation is not None:
            self.pool.rollback(txn._reservation)

        txn._state = BlockTxnState.ROLLED_BACK
        self._open_txn = None


    def _fail(self, txn: BlockPlanTxn) -> None:
        if txn.state not in (BlockTxnState.DISPATCHED, BlockTxnState.COMMITTED):
            raise ValueError("KV transaction must be dispatched or committed")
        self.teardown(txn)


    def _require_open(
        self,
        txn: BlockPlanTxn,
        expected_state: BlockTxnState,
    ) -> None:
        if (
            self._open_txn is not txn
            or txn.state is not expected_state
        ):
            raise ValueError(
                "KV transaction is not in the expected state"
            )


    def _blocks_for_tokens(self, token_count: int) -> int:
        return (
            token_count + self.block_size - 1
        ) // self.block_size


    def _slot_for_position(
        self,
        table: tuple[BlockRef, ...],
        position: int,
    ) -> int:
        # Canonical linear CPU slot. The backend may lower this
        # into its private tensor layout later.
        logical_block, offset = divmod(
            position,
            self.block_size,
        )
        return (
            table[logical_block].block_id
            * self.block_size
            + offset
        )
