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