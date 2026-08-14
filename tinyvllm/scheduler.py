from __future__ import annotations

from dataclasses import dataclass

from tinyvllm.request import (
    RequestState,
    RequestStatus,
    SamplingParams,
)


def _check_non_negative_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class AddCommand:
    seq: int
    request_id: int
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingParams
    arrival_ns: int


    def __post_int__(self) -> None:
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


class Scheduler:
    def __init__(self) -> None:
        self._requests: dict[int, RequestState] = {}
        self._waiting_ids: list[int] = []
        self._immediate_finished_ids: list[int] = []

        self._last_command_seq = -1
        self._now_ns: int | None = None


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