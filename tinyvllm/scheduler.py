from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from tinyvllm.batch import StepPlan, WorkItem
from tinyvllm.request import (
    RequestState,
    RequestStatus,
    SamplingParams,
    WorkClass,
)

from tinyvllm._validation import (
    _check_non_negative_int,
    _check_positive_int,
)


@dataclass(frozen=True, slots=True)
class AddCommand:
    seq: int
    request_id: int
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingParams
    arrival_ns: int


    def __post_init__(self) -> None:
        _check_non_negative_int("seq", self.seq)
        _check_non_negative_int("request_id", self.request_id)
        _check_non_negative_int("arrival_ns", self.arrival_ns)

        if not isinstance(self.sampling, SamplingParams):
            raise TypeError("sampling must be SamplingParams")

        object.__setattr__(
            self,
            "prompt_token_ids",
            tuple(self.prompt_token_ids),
        )


class TxnState(Enum):
    PREPARED = auto()
    DISPATCHED = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()
    FAILED = auto()


@dataclass(frozen=True, slots=True)
class StagedAdmission:
    request_id: int
    request_slot: int

    def __post_init__(self) -> None:
        _check_non_negative_int("request_id", self.request_id)
        _check_non_negative_int("request_slot", self.request_slot)


@dataclass(slots=True)
class ScheduleTxn:
    plan: StepPlan
    admissions: tuple[StagedAdmission, ...]
    state: TxnState = field(default=TxnState.PREPARED, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, StepPlan):
            raise TypeError("plan must be StepPlan")

        self.admissions = tuple(self.admissions)
        for admission in self.admissions:
            if not isinstance(admission, StagedAdmission):
                raise TypeError(
                    "admissions must contain StagedAdmission values"
                )

        plan_pairs = {
            (item.request_id, item.request_slot)
            for item in self.plan.items
        }
        admission_pairs = {
            (item.request_id, item.request_slot)
            for item in self.admissions
        }

        if len(admission_pairs) != len(self.admissions):
            raise ValueError("admissions must be unique")
        if not admission_pairs.issubset(plan_pairs):
            raise ValueError(
                "every admission must refer to a planned item"
            )


class Scheduler:
    def __init__(
        self,
        *,
        max_active_requests: int,
        max_step_requests: int,
        max_scheduled_tokens: int,
        prefill_chunk_size: int,
    ) -> None:
        _check_positive_int("max_active_requests", max_active_requests)
        _check_positive_int("max_step_requests", max_step_requests)
        _check_positive_int("max_scheduled_tokens", max_scheduled_tokens)
        _check_positive_int("prefill_chunk_size", prefill_chunk_size)

        if max_step_requests > max_active_requests:
            raise ValueError(
                "max_step_requests exceeds active-request capacity"
            )
        if prefill_chunk_size > max_scheduled_tokens:
            raise ValueError(
                "prefill_chunk_size exceeds the step token budget"
            )

        self.max_active_requests = max_active_requests
        self.max_step_requests = max_step_requests
        self.max_scheduled_tokens = max_scheduled_tokens
        self.prefill_chunk_size = prefill_chunk_size
        
        self._requests: dict[int, RequestState] = {}
        self._waiting_ids: deque[int] = deque()
        self._slot_request_ids: list[int | None] = (
            [None] * max_active_requests
        )
        self._immediate_finished_ids: list[int] = []

        self._last_command_seq = -1
        self._now_ns: int | None = None
        self._command_cutoff_ready = False
        self._next_plan_id = 0
        self._open_txn: ScheduleTxn | None = None


    @property
    def has_open_transaction(self) -> bool:
        return self._open_txn is not None


    @property
    def request_ids(self) -> tuple[int, ...]:
        return tuple(self._requests)


    @property
    def waiting_request_ids(self) -> tuple[int, ...]:
        return tuple(self._waiting_ids)


    @property
    def last_command_seq(self) -> int:
        return self._last_command_seq


    def get_request(self, request_id: int) -> RequestState:
        return self._requests[request_id]


    def take_immediate_finished_request_ids(self) -> tuple[int, ...]:
        requests_ids = tuple(self._immediate_finished_ids)
        self._immediate_finished_ids.clear()
        return requests_ids


    def apply_commands(
        self,
        commands: tuple[AddCommand, ...],
        now_ns: int,
    ) -> None:
        if self._open_txn is not None:
            raise RuntimeError(
                "cannot apply commands during an open transaction"
            )

        self._check_now(now_ns)

        commands = tuple(commands)
        expected_seq = self._last_command_seq + 1

        staged: list[RequestState] = []
        staged_request_ids: set[int] = set()

        for command in commands:
            if not isinstance(command, AddCommand):
                raise TypeError(
                    "commands must contain AddCommand values"
                )

            if command.seq != expected_seq:
                raise ValueError(
                    f"expected command seq {expected_seq}, "
                    f"got {command.seq}"
                )

            if (
                command.request_id in self._requests
                or command.request_id in staged_request_ids
            ):
                raise ValueError(
                    f"duplicate request_id: {command.request_id}"
                )

            request = RequestState(
                request_id=command.request_id,
                submit_seq=command.seq,
                prompt_token_ids=list(command.prompt_token_ids),
                sampling=command.sampling,
                arrival_ns=command.arrival_ns,
            )

            staged.append(request)
            staged_request_ids.add(command.request_id)
            expected_seq += 1

        # Publish only after the complete command batch validates.
        for request in staged:
            self._requests[request.request_id] = request

        if staged:
            self._last_command_seq = staged[-1].submit_seq

        self._now_ns = now_ns
        self._promote_arrivals(now_ns)
        self._command_cutoff_ready = True


    def begin_step(self, now_ns: int) -> ScheduleTxn | None:
        _check_non_negative_int("now_ns", now_ns)

        if self._open_txn is not None:
            raise RuntimeError(
                "a scheduling transaction is already open"
            )
        if (
            not self._command_cutoff_ready
            or self._now_ns is None
            or now_ns != self._now_ns
        ):
            raise ValueError(
                "apply_commands must run first with the same now_ns"
            )

        free_slots = self._free_slots()
        if not free_slots or not self._waiting_ids:
            return None

        remaining_tokens = self.max_scheduled_tokens
        items: list[WorkItem] = []
        admissions: list[StagedAdmission] = []

        for request_id, request_slot in zip(
            self._waiting_ids,
            free_slots[: self.max_step_requests],
        ):
            if remaining_tokens == 0:
                break
            request = self._requests[request_id]
            request.check_committed_invariants()

            if request.status is not RequestStatus.WAITING:
                raise RuntimeError(
                    "waiting queue contains a non-waiting request"
                )

            query_len = min(
                request.num_uncomputed_tokens,
                self.prefill_chunk_size,
                remaining_tokens,
            )
            if query_len <= 0:
                raise RuntimeError(
                    "waiting request has no schedulable tokens"
                )

            token_start = request.computed_tokens
            token_end = token_start + query_len

            items.append(
                WorkItem(
                    request_id=request.request_id,
                    request_slot=request_slot,
                    work_class=WorkClass.PREFILL,
                    token_start=token_start,
                    query_len=query_len,
                    context_len_before=token_start,
                    should_sample=token_end == request.total_tokens,
                )
            )
            admissions.append(
                StagedAdmission(
                    request_id=request.request_id,
                    request_slot=request_slot,
                )
            )
            remaining_tokens -= query_len

        if not items:
            return None

        txn = ScheduleTxn(
            plan=StepPlan(
                plan_id=self._next_plan_id,
                items=tuple(items),
            ),
            admissions=tuple(admissions),
        )
        self._next_plan_id += 1
        self._open_txn = txn
        return txn


    def mark_dispatched(
        self,
        txn: ScheduleTxn,
        now_ns: int,
    ) -> None:
        """Publish runtime ownership immediately before execution."""
        self._require_open_txn(txn, TxnState.PREPARED)
        if not self._command_cutoff_ready:
            raise RuntimeError("no command cutoff is ready")

        self._check_now(now_ns)

        admission_slots: dict[int, int] = {}
        for admission in txn.admissions:
            if not isinstance(admission, StagedAdmission):
                raise TypeError(
                    "admission must contain StagedAdmission values"
                )
            if admission.request_id in admission_slots:
                raise RuntimeError(
                    "transaction contains duplicate admissions"
                )
            admission_slots[admission.request_id] = (
                admission.request_slot
            )

        if len(self._waiting_ids) < len(txn.admissions):
            raise RuntimeError(
                "admission no longer match the waiting queue"
            )

        for waiting_id, admission in zip(
            self._waiting_ids,
            txn.admissions,
        ):
            if waiting_id != admission.request_id:
                raise RuntimeError(
                    "admission no longer match the waiting queue head"
                )

        plan_items = {
            item.request_id: item
            for item in txn.plan.items
        }

        for admission in txn.admissions:
            item = plan_items.get(admission.request_id)
            if (
                item is None
                or item.request_slot != admission.request_slot
            ):
                raise RuntimeError(
                    "admission does not match the step plan"
                )

        # Validate the complete plan before publishing any mutation.
        validated: list[tuple[RequestState, int | None]] = []

        for item in txn.plan.items:
            request = self._requests.get(item.request_id)
            if request is None:
                raise RuntimeError(
                    "plan refers to an unknown request"
                )
            if item.request_slot >= self.max_active_requests:
                raise RuntimeError(
                    "planned request slot is out of range"
                )
            if request.inflight_plan_id is not None:
                raise RuntimeError(
                    "planned request is already inflight"
                )
            if request.work_class is not item.work_class:
                raise RuntimeError(
                    "request work class no longer matches the plan"
                )
            if (
                request.computed_tokens != item.token_start
                or item.context_len_before != request.computed_tokens
            ):
                raise RuntimeError(
                    "request progress no longer matches the plan"
                )

            admission_slot = admission_slots.get(item.request_id)

            if admission_slot is not None:
                if request.status is not RequestStatus.WAITING:
                    raise RuntimeError(
                        "admitted request is not waiting"
                    )
                if request.active_slot is not None:
                    raise RuntimeError(
                        "admitted request already owns a slot"
                    )
                if self._slot_request_ids[item.request_slot] is not None:
                    raise RuntimeError(
                        "admission request slot is not free"
                    )
            else:
                if request.status is not RequestStatus.RUNNING:
                    raise RuntimeError(
                        "non-admitted plan item is not running"
                    )
                if request.active_slot != item.request_slot:
                    raise RuntimeError(
                        "running request slot no longer matches the plan"
                    )
                if (
                    self._slot_request_ids[item.request_slot]
                    != request.request_id
                ):
                    raise RuntimeError(
                        "running request does not own its slot"
                    )

            validated.append((request, admission_slot))

        # Publication starts here. Everything above must remain mutation-free.
        for _ in txn.admissions:
            self._waiting_ids.popleft()

        for request, admission_slot in validated:
            if admission_slot is not None:
                request.status = RequestStatus.RUNNING
                request.active_slot = admission_slot
                self._slot_request_ids[admission_slot] = (
                    request.request_id
                )

                if request.admitted_ns is None:
                    request.admitted_ns = now_ns

            request.inflight_plan_id = txn.plan.plan_id

        self._now_ns = now_ns
        self._command_cutoff_ready = False
        txn.state = TxnState.DISPATCHED


    def rollback(self, txn: ScheduleTxn) -> None:
        self._require_open_txn(txn, TxnState.PREPARED)

        txn.state = TxnState.ROLLED_BACK
        self._open_txn = None


    def _require_open_txn(
        self,
        txn: ScheduleTxn,
        expected_state: TxnState,
    ) -> None:
        if txn is not self._open_txn:
            raise ValueError(
                "transaction is not the scheduler's open transaction"
            )
        if txn.state is not expected_state:
            raise RuntimeError(
                f"transaction must be {expected_state.name.lower()}"
            )

    def _free_slots(self) -> list[int]:
        return [
            slot 
            for slot, request_id in enumerate(self._slot_request_ids)
            if request_id is None
        ]


    def _promote_arrivals(self, now_ns: int) -> None:
        eligible = sorted(
            (
                request
                for request in self._requests.values()
                if request.status is RequestStatus.PENDING
                and request.arrival_ns <= now_ns
            ),
            key=lambda request: (
                request.arrival_ns,
                request.submit_seq,
            ),
        )


        for request in eligible:
            if request.sampling.max_new_tokens == 0:
                request.status = RequestStatus.FINISHED
                request.admitted_ns = now_ns
                request.finished_ns = now_ns

                self._immediate_finished_ids.append(
                    request.request_id
                )
            else:
                request.status = RequestStatus.WAITING
                self._waiting_ids.append(request.request_id)

            request.check_committed_invariants()


    def _check_now(self, now_ns: int) -> None:
        _check_non_negative_int("now_ns", now_ns)

        if self._now_ns is not None and now_ns < self._now_ns:
            raise ValueError("now_ns cannot move backwards")
