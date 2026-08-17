from dataclasses import dataclass, field
from enum import Enum, auto

from tinyvllm.batch import StepPlan, WorkItem
from tinyvllm.request import (
    RequestState,
    RequestStatus,
    SamplingParams,
    WorkClass,
)


def _check_non_negative_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _check_positive_int(name: str, value: int) -> None:
    _check_non_negative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


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
        self._waiting_ids: list[int] = []
        self._immediate_finished_ids: list[int] = []

        self._last_command_seq = -1
        self._now_ns: int | None = None
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


    def begin_step(self, now_ns: int) -> ScheduleTxn | None:
        _check_non_negative_int("now_ns", now_ns)

        if self._open_txn is not None:
            raise RuntimeError(
                "a scheduling transaction is already open"
            )
        if self._now_ns is None or now_ns != self._now_ns:
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


    def rollback(self, txn: ScheduleTxn) -> None:
        if txn is not self._open_txn:
            raise ValueError(
                "transaction is not the scheduler's open transaction"
            )
        if txn.state is not TxnState.PREPARED:
            raise RuntimeError(
                "only a prepared transaction can roll back"
            )

        txn.state = TxnState.ROLLED_BACK
        self._open_txn = None

    def _free_slots(self) -> list[int]:
        used_slots: set[int] = set()

        for request in self._requests.values():
            if request.active_slot is None:
                continue
            if request.active_slot >= self.max_active_requests:
                raise RuntimeError(
                    "request owns an out-of-range active slot"
                )
            if request.active_slot in used_slots:
                raise RuntimeError(
                    "multiple requests own the same active slot"
                )
            used_slots.add(request.active_slot)

        return [slot for slot in range(self.max_active_requests)
                if slot not in used_slots]


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