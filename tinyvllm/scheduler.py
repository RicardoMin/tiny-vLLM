from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from heapq import heappop, heappush

from tinyvllm.batch import StepPlan, StepResult, WorkItem
from tinyvllm.request import (
    RequestState,
    RequestStatus,
    SamplingParams,
    WorkClass,
)

from tinyvllm._validation import (
    _check_non_negative_int,
    _check_positive_int,
    _check_token_id,
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


@dataclass(slots=True)
class ScheduleTxn:
    plan: StepPlan
    state: TxnState = field(default=TxnState.PREPARED, init=False)


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
        self._pending_arrivals: list[tuple[int, int, int]] = []
        self._waiting_ids: deque[int] = deque()
        self._prefill_ids: deque[int] = deque()
        self._decode_ids: deque[int] = deque()
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
            heappush(
                self._pending_arrivals,
                (
                    request.arrival_ns,
                    request.submit_seq,
                    request.request_id,
                ),
            )

        if staged:
            self._last_command_seq = staged[-1].submit_seq

        self._now_ns = now_ns
        self._promote_arrivals(now_ns)
        self._command_cutoff_ready = True


    def begin_step(self, now_ns: int) -> ScheduleTxn | None:
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

        remaining_rows = self.max_step_requests
        remaining_tokens = self.max_scheduled_tokens
        items: list[WorkItem] = []

        for request, request_slot, work_class in (
            self._work_candidates()
        ):
            if remaining_rows == 0 or remaining_tokens == 0:
                break

            query_len = (
                1
                if work_class is WorkClass.DECODE
                else min(
                    request.num_uncomputed_tokens,
                    self.prefill_chunk_size,
                    remaining_tokens,
                )
            )
            token_start = request.computed_tokens
            token_end = token_start + query_len

            items.append(
                WorkItem(
                    request_id=request.request_id,
                    request_slot=request_slot,
                    work_class=work_class,
                    token_start=token_start,
                    query_len=query_len,
                    context_len_before=token_start,
                    should_sample=(
                        work_class is WorkClass.DECODE
                        or token_end == request.total_tokens
                    ),
                )
            )
            remaining_rows -= 1
            remaining_tokens -= query_len

        if not items:
            return None

        txn = ScheduleTxn(
            plan=StepPlan(
                plan_id=self._next_plan_id,
                items=tuple(items),
            ),
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
        self._check_now(now_ns)

        plan_id = txn.plan.plan_id
        for item in txn.plan.items:
            request = self._requests[item.request_id]

            if request.status is RequestStatus.WAITING:
                self._waiting_ids.popleft()
                request.status = RequestStatus.RUNNING
                request.active_slot = item.request_slot
                self._slot_request_ids[item.request_slot] = (
                    request.request_id
                )

                if request.admitted_ns is None:
                    request.admitted_ns = now_ns

            self._prefill_ids.append(request.request_id)

            request.inflight_plan_id = plan_id

        self._now_ns = now_ns
        self._command_cutoff_ready = False
        txn.state = TxnState.DISPATCHED


    def rollback(self, txn: ScheduleTxn) -> None:
        self._require_open_txn(txn, TxnState.PREPARED)

        txn.state = TxnState.ROLLED_BACK
        self._open_txn = None


    def commit_step(
        self,
        txn: ScheduleTxn,
        result: StepResult,
        now_ns: int,
    ) -> tuple[int, ...]:
        self._require_open_txn(txn, TxnState.DISPATCHED)

        try:
            self._check_now(now_ns)
            plan = txn.plan

            if result.plan_id != plan.plan_id:
                raise RuntimeError("result plan_id does not match plan")
            if len(result.sample_request_ids) != plan.num_samples:
                raise RuntimeError("result sample count does not match plan")

            # Validate executor output before publishing any request state.
            sample_index = 0
            for item in plan.items:
                if not item.should_sample:
                    continue
                if result.sample_request_ids[sample_index] != item.request_id:
                    raise RuntimeError("result sampling order does not match plan")
                _check_token_id(result.sampled_token_ids[sample_index])
                sample_index += 1

            sample_index = 0
            finished_ids: list[int] = []

            for item in plan.items:
                request = self._requests[item.request_id]
                request.computed_tokens = item.token_end
                request.inflight_plan_id = None

                if item.work_class is WorkClass.PREFILL:
                    self._prefill_ids.popleft()
                else:
                    self._decode_ids.popleft()

                if not item.should_sample:
                    self._prefill_ids.append(request.request_id)
                    continue

                request.output_token_ids.append(
                    result.sampled_token_ids[sample_index]
                )
                sample_index += 1

                if request.first_token_ns is None:
                    request.first_token_ns = now_ns

                request.work_class = WorkClass.DECODE

                if request.remaining_new_tokens == 0:
                    self._slot_request_ids[item.request_slot] = None
                    request.active_slot = None
                    request.status = RequestStatus.FINISHED
                    request.finished_ns = now_ns
                    finished_ids.append(request.request_id)
                else:
                    self._decode_ids.append(request.request_id)

        except Exception:
            self.fail_step(txn)
            raise

        self._now_ns = now_ns
        txn.state = TxnState.COMMITTED
        self._open_txn = None
        return tuple(finished_ids)


    def fail_step(self, txn: ScheduleTxn) -> None:
        self._require_open_txn(txn, TxnState.DISPATCHED)
        txn.state = TxnState.FAILED


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


    def _work_candidates(self):
        for request_id in self._decode_ids:
            request = self._requests[request_id]
            yield request, request.active_slot, WorkClass.DECODE

        for request_id in self._prefill_ids:
            request = self._requests[request_id]
            yield request, request.active_slot, WorkClass.PREFILL

        if not self._waiting_ids:
            return

        for request_id, request_slot in zip(
            self._waiting_ids,
            self._free_slots(),
        ):
            yield (
                self._requests[request_id],
                request_slot,
                WorkClass.PREFILL
            )


    def _promote_arrivals(self, now_ns: int) -> None:
        while (
            self._pending_arrivals
            and self._pending_arrivals[0][0] <= now_ns
        ):
            _, _, request_id = heappop(self._pending_arrivals)
            request = self._requests[request_id]

            if request.status is not RequestStatus.PENDING:
                continue

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


    def _check_now(self, now_ns: int) -> None:
        _check_non_negative_int("now_ns", now_ns)

        if self._now_ns is not None and now_ns < self._now_ns:
            raise ValueError("now_ns cannot move backwards")
