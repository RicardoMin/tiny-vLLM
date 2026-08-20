from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from tinyvllm._validation import (
    _check_non_negative_int,
    _check_positive_int,
)


class PageState(Enum):
    FREE = auto()
    RESERVED = auto()
    PRIVATE_MUTABLE = auto()


@dataclass(frozen=True, slots=True)
class PageRef:
    page_id: int
    generation: int

    def __post_init__(self) -> None:
        _check_non_negative_int("page_id", self.page_id)
        _check_non_negative_int("generation", self.generation)


@dataclass(frozen=True, slots=True)
class PageReservation:
    txn_id: int
    page_refs: tuple[PageRef, ...]

    def __post_init__(self) -> None:
        _check_non_negative_int("txn_id", self.txn_id)
        object.__setattr__(self, "page_refs", tuple(self.page_refs))


@dataclass(frozen=True, slots=True)
class KvAppend:
    request_id: int
    token_start: int
    query_len: int

    def __post_init__(self) -> None:
        _check_non_negative_int("request_id", self.request_id)
        _check_non_negative_int("token_start", self.token_start)
        _check_positive_int("query_len", self.query_len)


class PageAllocator:
    """CPU ownership table for physical KV pages."""

    def __init__(self, *, num_pages: int) -> None:
        _check_positive_int("num_pages", num_pages)

        self._generations = [0] * num_pages
        self._states = [PageState.FREE] * num_pages

        # The last element is popped first.
        self._free_ids = list(range(num_pages - 1, -1, -1))

        self._open_reservations: dict[int, PageReservation] = {}

    @property
    def num_pages(self) -> int:
        return len(self._states)

    @property
    def free_page_count(self) -> int:
        return len(self._free_ids)

    @property
    def open_reservation_count(self) -> int:
        return len(self._open_reservations)


    def snapshot(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[PageState, ...]]:
        """Test-only deterministic state view."""
        return (
            tuple(self._free_ids),
            tuple(self._generations),
            tuple(self._states),
        )


    def reserve(
        self,
        count: int,
        txn_id: int,
    ) -> PageReservation | None:
        _check_non_negative_int("count", count)
        _check_non_negative_int("txn_id", txn_id)

        if txn_id in self._open_reservations:
            raise ValueError(
                f"transaction already owns a reservation: {txn_id}"
            )

        # This check happens before mutating any state.
        if count > self.free_page_count:
            return None

        page_refs: list[PageRef] = []

        for _ in range(count):
            page_id = self._free_ids.pop()

            # Increment on allocation, including after rollback.
            generation = self._generations[page_id] + 1
            self._generations[page_id] = generation
            self._states[page_id] = PageState.RESERVED

            page_refs.append(PageRef(page_id, generation))

        reservation = PageReservation(
            txn_id=txn_id,
            page_refs=tuple(page_refs),
        )
        self._open_reservations[txn_id] = reservation
        return reservation


    def commit(self, reservation: PageReservation) -> None:
        self._validate_open_reservation(reservation)
        self._validate_reserved_pages(reservation)

        for ref in reservation.page_refs:
            self._states[ref.page_id] = PageState.PRIVATE_MUTABLE

        del self._open_reservations[reservation.txn_id]


    def rollback(self, reservation: PageReservation) -> None:
        self._validate_open_reservation(reservation)
        self._validate_reserved_pages(reservation)

        for ref in reservation.page_refs:
            self._states[ref.page_id] = PageState.FREE

        # reserve() popped page IDs in page_refs order.
        # Reverse them to restore the original stack order.
        self._free_ids.extend(
            reversed(
                tuple(ref.page_id for ref in reservation.page_refs)
            )
        )

        # Generation is intentionally not rolled back.
        del self._open_reservations[reservation.txn_id]


    def release(self, ref: PageRef) -> None:
        self._validate_ref(ref)

        if self._states[ref.page_id] is not PageState.PRIVATE_MUTABLE:
            raise RuntimeError(
                "only private mutable pages can be released"
            )

        self._states[ref.page_id] = PageState.FREE
        self._free_ids.append(ref.page_id)


    def page_state(self, ref: PageRef) -> PageState:
        self._validate_ref(ref)
        return self._states[ref.page_id]

    def _validate_open_reservation(
        self,
        reservation: PageReservation,
    ) -> None:
        active = self._open_reservations.get(reservation.txn_id)

        # Identity checking prevents an externally fabricated equivalent
        # reservation from controlling the allocator.
        if active is not reservation:
            raise ValueError(
                "reservation is not open in this allocator"
            )


    def _validate_reserved_pages(
        self,
        reservation: PageReservation,
    ) -> None:
        for ref in reservation.page_refs:
            self._validate_ref(ref)

            if self._states[ref.page_id] is not PageState.RESERVED:
                raise RuntimeError(
                    "reservation contains a non-reserved page"
                )


    def _validate_ref(self, ref: PageRef) -> None:
        if not isinstance(ref, PageRef):
            raise TypeError("ref must be PageRef")

        if ref.page_id >= self.num_pages:
            raise ValueError("page_id is out of range")

        if self._generations[ref.page_id] != ref.generation:
            raise RuntimeError("stale PageRef generation")



class KvTxnState(Enum):
    PREPARED = auto()
    DISPATCHED = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()


class KvPlanTxn:
    """One step's staged KV page-table changes."""

    __slots__ = (
        "_plan_id",
        "_state",
        "_manager",
        "_reservation",
        "_page_tables",
        "_slot_mappings",
        "_token_ends",
        "_written_refs",
    )


    def __init__(
        self,
        *,
        plan_id: int,
        manager: KvPageManager,
        reservation: PageReservation | None,
        page_tables: dict[int, tuple[PageRef, ...]],
        slot_mappings: dict[int, tuple[int, ...]],
        token_ends: dict[int, int],
        written_refs: tuple[PageRef, ...],
    ) -> None:
        self._plan_id = plan_id
        self._state = KvTxnState.PREPARED
        self._manager = manager
        self._reservation = reservation
        self._page_tables = page_tables
        self._slot_mappings = slot_mappings
        self._token_ends = token_ends
        self._written_refs = written_refs


    @property
    def state(self) -> KvTxnState:
        return self._state

    @property
    def plan_id(self) -> int:
        return self._plan_id

    def staged_page_table(
        self,
        request_id: int,
    ) -> tuple[PageRef, ...]:
        return self._page_tables[request_id]

    def slot_mapping(self, request_id: int) -> tuple[int, ...]:
        return self._slot_mappings[request_id]

    def commit_before_dispatch(self) -> None:
        self._manager._commit_before_dispatch(self)

    def complete_writes(self) -> tuple[PageRef, ...]:
        return self._manager._complete_writes(self)

    def rollback(self) -> None:
        self._manager._rollback(self)


class KvPageManager:
    """Owns request page tables and stages append work transactionally"""

    def __init__(
        self,
        *,
        page_size: int,
        num_pages: int,
    ) -> None:
        _check_positive_int("page_size", page_size)

        self.page_size = page_size
        self.allocator = PageAllocator(num_pages=num_pages)

        self._page_tables: dict[int, tuple[PageRef, ...]] = {}
        self._valid_token_counts: dict[int, int] = {}
        self._open_txn: KvPlanTxn | None = None


    def page_table(
        self,
        request_id: int,
    ) -> tuple[PageRef, ...]:
        return self._page_tables.get(request_id, ())


    def valid_token_count(self, request_id: int) -> int:
        return self._valid_token_counts.get(request_id, 0)


    def begin_plan(
        self,
        *,
        plan_id: int,
        appends: tuple[KvAppend, ...],
    ) -> KvPlanTxn | None:
        """Stage request append intervals as one KV transaction."""
        _check_non_negative_int("plan_id", plan_id)

        if self._open_txn is not None:
            raise RuntimeError("a KV transaction is already open")

        appends = tuple(appends)
        if not appends:
            raise ValueError("KV plan must contain at least one append")

        staged: list[
            tuple[int, int, tuple[PageRef, ...], int, int]
        ] = []
        seen_request_ids: set[int] = set()
        total_new_pages = 0

        # Validate the complete batch before reserving any physical page.
        for append in appends:
            if not isinstance(append, KvAppend):
                raise TypeError(
                    "appends must contain KvAppend values"
                )

            request_id = append.request_id
            token_start = append.token_start
            query_len = append.query_len

            if request_id in seen_request_ids:
                raise ValueError(
                    "request appears twice in one KV plan"
                )
            seen_request_ids.add(request_id)

            valid_tokens = self.valid_token_count(request_id)
            if token_start != valid_tokens:
                raise ValueError(
                    "token_start does not match valid KV length"
                )

            table = self.page_table(request_id)
            token_end = token_start + query_len
            required_pages = self._pages_for_tokens(token_end)
            new_page_count = required_pages - len(table)

            if new_page_count < 0:
                raise RuntimeError(
                    "page table exceeds valid KV length"
                )

            staged.append(
                (
                    request_id,
                    token_start,
                    table,
                    token_end,
                    new_page_count,
                )
            )
            total_new_pages += new_page_count

        reservation = None
        if total_new_pages:
            reservation = self.allocator.reserve(
                count=total_new_pages,
                txn_id=plan_id,
            )
            if reservation is None:
                return None

        reserved_refs = (
            reservation.page_refs
            if reservation is not None
            else ()
        )

        ref_offset = 0
        page_tables: dict[int, tuple[PageRef, ...]] = {}
        slot_mappings: dict[int, tuple[int, ...]] = {}
        token_ends: dict[int, int] = {}
        written_refs: list[PageRef] = []

        for (
            request_id,
            token_start,
            table,
            token_end,
            new_page_count,
        ) in staged:
            new_refs = reserved_refs[
                ref_offset : ref_offset + new_page_count
            ]
            ref_offset += new_page_count

            staged_table = table + new_refs
            page_tables[request_id] = staged_table
            token_ends[request_id] = token_end

            first_page = token_start // self.page_size
            last_page = token_end // self.page_size

            written_refs.extend(
                staged_table[first_page : last_page + 1]
            )

            slot_mappings[request_id] = tuple(
                self._slot_for_position(
                    staged_table,
                    position,
                )
                for position in range(token_start, token_end)
            )

        txn = KvPlanTxn(
            plan_id=plan_id,
            manager=self,
            reservation=reservation,
            page_tables=page_tables,
            slot_mappings=slot_mappings,
            token_ends=token_ends,
            written_refs=tuple(written_refs),
        )

        self._open_txn = txn
        return txn


    def release_request(self, request_id: int) -> None:
        if self._open_txn is not None:
            raise RuntimeError(
                "cannot release pages during a KV transaction"
            )

        table = self._page_tables.pop(request_id, ())
        self._valid_token_counts.pop(request_id, None)

        for ref in reversed(table):
            self.allocator.release(ref)


    def _commit_before_dispatch(self, txn: KvPlanTxn) -> None:
        self._require_open(txn, KvTxnState.PREPARED)

        if txn._reservation is not None:
            self.allocator.commit(txn._reservation)

        self._page_tables.update(txn._page_tables)
        txn._state = KvTxnState.DISPATCHED


    def _complete_writes(
        self,
        txn: KvPlanTxn,
    ) -> tuple[PageRef, ...]:
        self._require_open(txn, KvTxnState.DISPATCHED)

        self._valid_token_counts.update(txn._token_ends)
        txn._state = KvTxnState.COMMITTED
        self._open_txn = None

        return txn._written_refs


    def _rollback(self, txn: KvPlanTxn) -> None:
        self._require_open(txn, KvTxnState.PREPARED)

        if txn._reservation is not None:
            self.allocator.rollback(txn._reservation)

        txn._state = KvTxnState.ROLLED_BACK
        self._open_txn = None


    def _require_open(
        self,
        txn: KvPlanTxn,
        expected_state: KvTxnState,
    ) -> None:
        if (
            self._open_txn is not txn
            or txn.state is not expected_state
        ):
            raise ValueError(
                "KV transaction is not in the expected state"
            )


    def _pages_for_tokens(self, token_count: int) -> int:
        return (
            token_count + self.page_size - 1
        ) // self.page_size


    def _slot_for_position(
        self,
        table: tuple[PageRef, ...],
        position: int,
    ) -> int:
        # Canonical linear CPU slot. The backend may lower this
        # into its private tensor layout later.
        logical_page, offset = divmod(
            position,
            self.page_size,
        )
        return (
            table[logical_page].page_id
            * self.page_size
            + offset
        )