from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from heapq import heappop, heappush

from tinyvllm.batch import StepPlan, StepResult, WorkItem
from tinyvllm.request import (
    CompletionEvent,
    RequestState,
    RequestStatus,
    SamplingParams,
    WorkClass,
)

from tinyvllm._validation import (
    _check_non_negative_int,
    _check_positive_int,
    _check_token_id,
    _normalize_token_ids,
)

from tinyvllm.cache import (
    KvAppend,
    KvPageManager,
    KvPlanTxn,
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


@dataclass(frozen=True, slots=True)
class CancelCommand:
    seq: int
    request_id: int

    def __post_init__(self) -> None:
        _check_non_negative_int("seq", self.seq)
        _check_non_negative_int("request_id", self.request_id)


EngineCommand = AddCommand | CancelCommand

class TxnState(Enum):
    PREPARED = auto()
    DISPATCHED = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()
    FAILED = auto()


class ScheduleDecision(Enum):
    """Why the scheduler did or did not produce a step plan."""

    PLAN = auto()
    IDLE = auto()
    BLOCKED = auto()


class BlockedReason(Enum):
    """Resource condition that prevents the next plan from progressing."""

    KV_CAPACITY = auto()


class SchedulerResourceError(RuntimeError):
    """The scheduler cannot make progress with its configured resources."""

    def __init__(self, reason: BlockedReason | None) -> None:
        self.reason = reason
        detail = "unknown reason" if reason is None else reason.name.lower()
        super().__init__(f"scheduler is blocked: {detail}")


@dataclass(frozen=True, slots=True)
class ScheduleTxn:
    plan: StepPlan
    kv: KvPlanTxn
    rotate_prefill: bool = False
    kv_fallback: bool = False
    state: TxnState = field(default=TxnState.PREPARED, init=False)


class Scheduler:
    def __init__(
        self,
        *,
        max_active_requests: int,
        max_step_requests: int,
        max_scheduled_tokens: int,
        prefill_chunk_size: int,
        max_model_len: int,
        prefill_reserve_tokens: int = 128,
        prefill_reserve_requests: int = 1,
        kv_page_size: int = 16,
        kv_num_pages: int | None = None,
        eos_token_ids: Iterable[int] | None = None,
    ) -> None:
        _check_positive_int("max_active_requests", max_active_requests)
        _check_positive_int("max_step_requests", max_step_requests)
        _check_positive_int("max_scheduled_tokens", max_scheduled_tokens)
        _check_positive_int("prefill_chunk_size", prefill_chunk_size)
        _check_positive_int("max_model_len", max_model_len)
        _check_positive_int("kv_page_size", kv_page_size)
        _check_non_negative_int("prefill_reserve_tokens", prefill_reserve_tokens)
        _check_non_negative_int("prefill_reserve_requests", prefill_reserve_requests)

        if prefill_reserve_tokens < prefill_reserve_requests:
            raise ValueError(
                "each reserved prefill request needs one token"
            )
        if (prefill_reserve_tokens == 0) != (
            prefill_reserve_requests == 0
        ):
            raise ValueError(
                "prefill reservations must both be zero or positive"
            )

        if max_step_requests > max_active_requests:
            raise ValueError(
                "max_step_requests exceeds active-request capacity"
            )
        if prefill_chunk_size > max_scheduled_tokens:
            raise ValueError(
                "prefill_chunk_size exceeds the step token budget"
            )

        if kv_num_pages is None:
            kv_num_pages = max_active_requests * (
                (max_model_len + kv_page_size - 1) // kv_page_size
            )
        else:
            _check_positive_int("kv_num_pages", kv_num_pages)


        self.max_active_requests = max_active_requests
        self.max_step_requests = max_step_requests
        self.max_scheduled_tokens = max_scheduled_tokens
        self.prefill_chunk_size = prefill_chunk_size
        self.max_model_len = max_model_len
        self.prefill_reserve_tokens = prefill_reserve_tokens
        self.prefill_reserve_requests = prefill_reserve_requests
        self.kv_page_size = kv_page_size
        self._kv_manager = KvPageManager(
            page_size=kv_page_size,
            num_pages=kv_num_pages,
        )

        self._eos_token_ids = _normalize_token_ids("eos_token_ids", eos_token_ids) if eos_token_ids is not None else set()

        self._requests: dict[int, RequestState] = {}
        self._pending_arrivals: list[tuple[int, int, int]] = []
        self._waiting_ids: deque[int] = deque()
        self._prefill_ids: deque[int] = deque()
        self._decode_ids: deque[int] = deque()
        self._prefer_waiting_prefill = False
        self._slot_request_ids: list[int | None] = (
            [None] * max_active_requests
        )
        self._completion_events: deque[CompletionEvent] = deque()

        self._last_command_seq = -1
        self._now_ns: int | None = None
        self._command_cutoff_ready = False
        self._next_plan_id = 0
        self._open_txn: ScheduleTxn | None = None
        self._last_decision = ScheduleDecision.IDLE
        self._blocked_reason: BlockedReason | None = None
        self._failed = False


    @property
    def eos_token_ids(self) -> frozenset[int]:
        return self._eos_token_ids


    @property
    def has_open_transaction(self) -> bool:
        return self._open_txn is not None


    @property
    def last_decision(self) -> ScheduleDecision:
        return self._last_decision


    @property
    def blocked_reason(self) -> BlockedReason | None:
        return self._blocked_reason


    @property
    def request_ids(self) -> tuple[int, ...]:
        """Includes terminal requests whose results whose results are not consumed yet."""
        return tuple(self._requests)


    @property
    def waiting_request_ids(self) -> tuple[int, ...]:
        return tuple(self._waiting_ids)


    @property
    def last_command_seq(self) -> int:
        return self._last_command_seq


    @property
    def kv_manager(self) -> KvPageManager:
        return self._kv_manager


    @property
    def completed_request_ids(self) -> tuple[int, ...]:
        return tuple(
            event.request_id
            for event in self._completion_events
        )


    def get_request(self, request_id: int) -> RequestState:
        return self._requests[request_id]


    def consume_finished(
        self,
        max_items: int | None = None,
    ) -> tuple[CompletionEvent, ...]:
        if max_items is not None:
            _check_non_negative_int("max_items", max_items)

            if max_items == 0:
                return ()

        count = len(self._completion_events)

        if max_items is not None:
            count = min(count, max_items)

        events: list[CompletionEvent] = []

        for _ in range(count):
            event = self._completion_events.popleft()
            del self._requests[event.request_id]
            events.append(event)

        return tuple(events)


    def apply_commands(
        self,
        commands: tuple[EngineCommand, ...],
        now_ns: int,
    ) -> None:
        self._ensure_healthy()
        if self._open_txn is not None:
            raise RuntimeError(
                "cannot apply commands during an open transaction"
            )

        self._check_now(now_ns)

        commands = tuple(commands)
        expected_seq = self._last_command_seq + 1

        staged: list[RequestState] = []
        staged_request_ids: set[int] = set()
        cancellations: list[CancelCommand] = []

        for command in commands:
            if not isinstance(command, (AddCommand, CancelCommand)):
                raise TypeError(
                    "commands must contain AddCommand or CancelCommand"
                )

            if command.seq != expected_seq:
                raise ValueError(
                    f"expected command seq {expected_seq}, "
                    f"got {command.seq}"
                )

            if isinstance(command, AddCommand):
                if (
                    command.request_id in self._requests
                    or command.request_id in staged_request_ids
                ):
                    raise ValueError(
                        f"duplicate request_id: {command.request_id}"
                    )

                if len(command.prompt_token_ids) > self.max_model_len:
                    raise ValueError("prompt exceeds max_model_len")

                staged.append(
                    RequestState(
                        request_id=command.request_id,
                        submit_seq=command.seq,
                        prompt_token_ids=command.prompt_token_ids,
                        sampling=command.sampling,
                        arrival_ns=command.arrival_ns,
                    )
                )
                staged_request_ids.add(command.request_id)
            else:
                if (
                    command.request_id not in self._requests
                    and command.request_id not in staged_request_ids
                ):
                    raise KeyError(
                        f"unknow request_id: {command.request_id}"
                    )

                cancellations.append(command)

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

        for command in cancellations:
            self._abort_request(
                self._requests[command.request_id],
                now_ns,
            )

        if commands:
            self._last_command_seq = commands[-1].seq

        self._now_ns = now_ns
        self._promote_arrivals(now_ns)
        self._command_cutoff_ready = True


    def begin_step(self, now_ns: int) -> ScheduleTxn | None:
        self._ensure_healthy()
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

        self._last_decision = ScheduleDecision.IDLE
        self._blocked_reason = None

        decode_candidates = [
            (
                self._requests[request_id],
                self._requests[request_id].active_slot,
            )
            for request_id in self._decode_ids
        ]

        active_prefill_candidates = [
            (
                self._requests[request_id],
                self._requests[request_id].active_slot,
            )
            for request_id in self._prefill_ids
        ]

        waiting_prefill_candidates = []
        if self._waiting_ids:
            waiting_prefill_candidates.extend(
                (self._requests[request_id], request_slot)
                for request_id, request_slot in zip(
                    self._waiting_ids,
                    self._free_slots(),
                )
            )

        has_active_prefill = bool(active_prefill_candidates)
        has_waiting_prefill = bool(waiting_prefill_candidates)

        if self._prefer_waiting_prefill:
            prefill_candidates = waiting_prefill_candidates
            prefill_candidates.extend(active_prefill_candidates)
        else:
            prefill_candidates = active_prefill_candidates
            prefill_candidates.extend(waiting_prefill_candidates)

        reserve_rows = 0
        reserve_tokens = 0

        if decode_candidates and prefill_candidates:
            reserve_rows = min(
                self.prefill_reserve_requests,
                self.max_step_requests - 1,
                len(prefill_candidates),
            )
            reserve_tokens = min(
                self.prefill_reserve_tokens,
                self.max_scheduled_tokens - 1,
            )

            reserve_rows = min(
                reserve_rows,
                reserve_tokens,
            )

            if reserve_rows == 0:
                reserve_tokens = 0

        remaining_rows = self.max_step_requests
        remaining_tokens = self.max_scheduled_tokens

        decode_count = min(
            len(decode_candidates),
            remaining_rows - reserve_rows,
            remaining_tokens - reserve_tokens,
        )

        remaining_rows -= decode_count
        remaining_tokens -= decode_count

        prefill_budget = (
            remaining_tokens
            if decode_count == len(decode_candidates)
            else reserve_tokens
        )

        prefill_items, prefill_index, prefill_tokens = (
            self._take_prefills(
                prefill_candidates,
                start=0,
                row_budget=reserve_rows,
                token_budget=prefill_budget,
                spread=True,
            )
        )

        remaining_rows -= len(prefill_items)
        remaining_tokens -= prefill_tokens

        decode_count += min(
            len(decode_candidates) - decode_count,
            remaining_rows,
            remaining_tokens,
        )

        remaining_rows = self.max_step_requests - (
            decode_count + len(prefill_items)
        )
        remaining_tokens = self.max_scheduled_tokens - (
            decode_count + prefill_tokens
        )

        if decode_count == len(decode_candidates):
            extra_items, _, _ = self._take_prefills(
                prefill_candidates,
                start=prefill_index,
                row_budget=remaining_rows,
                token_budget=remaining_tokens,
                spread=False,
            )
            prefill_items.extend(extra_items)

        decode_items = [
            self._make_work_item(
                request,
                request_slot,
                WorkClass.DECODE,
                1,
            )
            for request, request_slot in (
                decode_candidates[:decode_count]
            )
        ]

        items = decode_items + prefill_items

        if not items:
            self._last_decision = ScheduleDecision.IDLE
            return None

        plan = StepPlan(
            plan_id=self._next_plan_id,
            items=tuple(items),
        )

        kv_txn = self._kv_manager.begin_plan(
            plan_id=plan.plan_id,
            appends=tuple(
                KvAppend(
                    request_id=item.request_id,
                    token_start=item.token_start,
                    query_len=item.query_len,
                )
                for item in plan.items
            ),
        )

        kv_fallback = kv_txn is None
        if kv_fallback:
            items = self._fit_kv_capacity(items)
            if not items:
                self._last_decision = ScheduleDecision.BLOCKED
                self._blocked_reason = BlockedReason.KV_CAPACITY
                return None

            plan = StepPlan(
                plan_id=plan.plan_id,
                items=tuple(items),
            )
            kv_txn = self._kv_manager.begin_plan(
                plan_id=plan.plan_id,
                appends=tuple(
                    KvAppend(
                        request_id=item.request_id,
                        token_start=item.token_start,
                        query_len=item.query_len,
                    )
                    for item in plan.items
                ),
            )
            if kv_txn is None:
                raise RuntimeError(
                    "KV fallback selected an over-capacity batch"
                )

        txn = ScheduleTxn(
            plan=plan,
            kv=kv_txn,
            rotate_prefill=(
                has_active_prefill and has_waiting_prefill
            ),
            kv_fallback=kv_fallback,
        )
        self._next_plan_id += 1
        self._open_txn = txn
        self._last_decision = ScheduleDecision.PLAN
        return txn


    def mark_dispatched(
        self,
        txn: ScheduleTxn,
        now_ns: int,
    ) -> None:
        """Publish runtime ownership immediately before execution."""
        self._require_open_txn(txn, TxnState.PREPARED)
        self._check_now(now_ns)

        txn.kv.commit_before_dispatch()
        object.__setattr__(
            txn,
            "state",
            TxnState.DISPATCHED,
        )

        try:
            plan_id = txn.plan.plan_id
            for item in txn.plan.items:
                request = self._requests[item.request_id]

                if item.work_class is WorkClass.DECODE:
                    self._pop_ready(
                        self._decode_ids,
                        request.request_id,
                        txn.kv_fallback,
                    )
                elif request.status is RequestStatus.WAITING:
                    self._pop_ready(
                        self._waiting_ids,
                        request.request_id,
                        txn.kv_fallback,
                    )
                    request.status = RequestStatus.RUNNING
                    request.active_slot = item.request_slot
                    self._slot_request_ids[item.request_slot] = (
                        request.request_id
                    )

                    if request.admitted_ns is None:
                        request.admitted_ns = now_ns
                else:
                    self._pop_ready(
                        self._prefill_ids,
                        request.request_id,
                        txn.kv_fallback,
                    )

                request.inflight_plan_id = plan_id

            self._now_ns = now_ns
            self._command_cutoff_ready = False
        except Exception:
            # KV already dispatch, could not rollback.
            self.fail_step(txn)
            raise


    def rollback(self, txn: ScheduleTxn) -> None:
        self._require_open_txn(txn, TxnState.PREPARED)

        txn.kv.rollback()

        object.__setattr__(txn, "state", TxnState.ROLLED_BACK)
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

            # Executor 结果验证完毕后，推进 KV 的有效 token 数.
            txn.kv.complete_writes()

            sample_index = 0
            finished_ids: list[int] = []

            for item in plan.items:
                request = self._requests[item.request_id]
                request.computed_tokens = item.token_end
                request.inflight_plan_id = None

                if not item.should_sample:
                    if request.computed_tokens >= self.max_model_len:
                        self._finish_request(request, now_ns)
                        finished_ids.append(request.request_id)
                    else:
                        self._prefill_ids.append(request.request_id)
                    continue

                token_id = result.sampled_token_ids[sample_index]
                request.output_token_ids.append(token_id)
                sample_index += 1

                if request.first_token_ns is None:
                    request.first_token_ns = now_ns

                request.work_class = WorkClass.DECODE

                stop_on_eos = (
                    token_id in self._eos_token_ids
                    and not request.sampling.ignore_eos
                )

                if (
                    stop_on_eos
                    or request.remaining_new_tokens == 0
                    or request.computed_tokens >= self.max_model_len
                ):
                    self._finish_request(
                        request,
                        now_ns,
                    )
                    finished_ids.append(request.request_id)
                else:
                    self._decode_ids.append(request.request_id)

            if txn.rotate_prefill and plan.num_prefill_items:
                self._prefer_waiting_prefill = (
                    not self._prefer_waiting_prefill
                )

        except Exception:
            self.fail_step(txn)
            raise

        self._now_ns = now_ns
        object.__setattr__(txn, "state", TxnState.COMMITTED)
        self._open_txn = None
        return tuple(finished_ids)


    def fail_step(self, txn: ScheduleTxn) -> None:
        self._require_open_txn(txn, TxnState.DISPATCHED)
        txn.kv.fail()
        object.__setattr__(txn, "state", TxnState.FAILED)
        for item in txn.plan.items:
            request = self._requests.get(item.request_id)
            if request is not None:
                request.inflight_plan_id = None
        self._waiting_ids.clear()
        self._prefill_ids.clear()
        self._decode_ids.clear()
        self._slot_request_ids[:] = [None] * self.max_active_requests
        self._open_txn = None
        self._failed = True


    def _ensure_healthy(self) -> None:
        if self._failed:
            raise RuntimeError("scheduler is failed")


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

    def _free_slots(self) -> Iterator[int]:
        return (
            slot
            for slot, request_id in enumerate(self._slot_request_ids)
            if request_id is None
        )

    @staticmethod
    def _pop_ready(
        ready_ids: deque[int],
        request_id: int,
        kv_fallback: bool,
    ) -> None:
        if kv_fallback:
            ready_ids.remove(request_id)
        else:
            ready_ids.popleft()


    def _max_prefill_tokens(
        self,
        item: WorkItem,
        remaining_pages: int,
    ) -> int:
        allocated_pages = len(
            self._kv_manager.page_table(item.request_id)
        )
        tail_capacity = (
            allocated_pages * self.kv_page_size
            - item.token_start
        )
        return max(
            0,
            tail_capacity + remaining_pages * self.kv_page_size,
        )

    def _fit_kv_capacity(
        self,
        items: list[WorkItem],
    ) -> list[WorkItem]:
        remaining_pages = self._kv_manager.free_page_count
        selected: list[WorkItem] = []

        for item in items:
            query_len = item.query_len

            if item.work_class is WorkClass.PREFILL:
                query_len = min(
                    query_len,
                    self._max_prefill_tokens(
                        item,
                        remaining_pages,
                    ),
                )

            if query_len <= 0:
                continue

            if query_len != item.query_len:
                request = self._requests[item.request_id]
                item = self._make_work_item(
                    request,
                    item.request_slot,
                    item.work_class,
                    query_len,
                )
            
            append = KvAppend(
                request_id=item.request_id,
                token_start=item.token_start,
                query_len=item.query_len,
            )
            new_pages = self._kv_manager.required_new_pages(append)

            if new_pages > remaining_pages:
                continue

            selected.append(item)
            remaining_pages -= new_pages

        if not selected:
            return []

        free_slots = self._free_slots()
        compacted: list[WorkItem] = []

        for item in selected:
            request = self._requests[item.request_id]

            if request.status is RequestStatus.WAITING:
                item = self._make_work_item(
                    request,
                    next(free_slots),
                    item.work_class,
                    item.query_len,
                )

            compacted.append(item)

        return compacted

    def _make_work_item(
        self,
        request: RequestState,
        request_slot: int,
        work_class: WorkClass,
        query_len: int,
    ) -> WorkItem:
        token_start = request.computed_tokens
        token_end = token_start + query_len

        return WorkItem(
            request_id=request.request_id,
            request_slot=request_slot,
            work_class=work_class,
            token_start=token_start,
            query_len=query_len,
            context_len_before=token_start,
            should_sample=(
                work_class is WorkClass.DECODE
                or (
                    token_end == request.total_tokens
                    and token_end < self.max_model_len
                )
            ),
        )

    def _take_prefills(
        self,
        candidates,
        *,
        start: int,
        row_budget,
        token_budget,
        spread: bool,
    ) -> tuple[list[WorkItem], int, int]:
        count = min(
            len(candidates) - start,
            row_budget,
            token_budget,
        )

        items: list[WorkItem] = []
        used_tokens = 0

        for offset in range(count):
            request, request_slot = candidates[start + offset]

            tokens_for_later = (
                count - offset - 1
                if spread
                else 0
            )

            query_len = min(
                request.num_uncomputed_tokens,
                self.prefill_chunk_size,
                token_budget
                - used_tokens
                - tokens_for_later,
            )

            if query_len == 0:
                break

            items.append(
                self._make_work_item(
                    request,
                    request_slot,
                    WorkClass.PREFILL,
                    query_len,
                )
            )
            used_tokens += query_len

        return (
            items,
            start + len(items),
            used_tokens,
        )


    def _abort_request(
        self,
        request: RequestState,
        now_ns: int,
    ) -> None:
        if request.terminal:
            return

        if request.inflight_plan_id is not None:
            request.cancel_requested = True
            return

        for queue in (
            self._waiting_ids,
            self._prefill_ids,
            self._decode_ids,
        ):
            try:
                queue.remove(request.request_id)
            except ValueError:
                pass

        if request.active_slot is not None:
            self._slot_request_ids[request.active_slot] = None
            request.active_slot = None

        self._terminalize_request(
            request,
            RequestStatus.ABORTED,
            now_ns,
        )


    def _terminalize_request(
        self,
        request: RequestState,
        status: RequestStatus,
        now_ns: int,
    ) -> None:
        request.status = status
        request.cancel_requested = False
        request.inflight_plan_id = None
        request.finished_ns = now_ns

        if request.active_slot is not None:
            self._slot_request_ids[request.active_slot] = None
            request.active_slot = None

        self._kv_manager.release_request(
            request.request_id
        )

        self._completion_events.append(
            CompletionEvent.from_request(request)
        )


    def _finish_request(
        self,
        request: RequestState,
        now_ns: int,
    ) -> None:
        self._terminalize_request(
            request,
            RequestStatus.FINISHED,
            now_ns,
        )

    def _promote_arrivals(self, now_ns: int) -> None:
        while (
            self._pending_arrivals
            and self._pending_arrivals[0][0] <= now_ns
        ):
            _, _, request_id = heappop(self._pending_arrivals)
            request = self._requests.get(request_id)

            if request is None:
                continue

            if request.status is not RequestStatus.PENDING:
                continue

            if request.sampling.max_new_tokens == 0:
                request.admitted_ns = now_ns
                self._finish_request(
                    request,
                    now_ns,
                )
            else:
                request.status = RequestStatus.WAITING
                self._waiting_ids.append(request.request_id)


    def _check_now(self, now_ns: int) -> None:
        _check_non_negative_int("now_ns", now_ns)

        if self._now_ns is not None and now_ns < self._now_ns:
            raise ValueError("now_ns cannot move backwards")
