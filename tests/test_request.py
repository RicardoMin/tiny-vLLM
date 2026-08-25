import pytest

from tinyvllm import RequestStatus, SamplingParams, WorkClass, RequestState


def test_sampling_modes() -> None:
    greedy = SamplingParams(temperature=0)

    assert greedy.temperature == 0.0
    assert greedy.greedy
    assert not SamplingParams(temperature=0.6).greedy


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_new_tokens": -1}, ValueError),
        ({"max_new_tokens": True}, TypeError),
        ({"temperature": float("nan")}, ValueError),
        ({"temperature": True}, TypeError),
        ({"ignore_eos": 1}, TypeError),
        ({"seed": -1}, ValueError),
    ],
)
def test_sampling_rejects_invalid_values(kwargs, error) -> None:
    with pytest.raises(error):
        SamplingParams(**kwargs)


def make_request(prompt: list[int] | None = None) -> RequestState:
    return RequestState(
        request_id=7,
        submit_seq=3,
        prompt_token_ids=[11, 12, 13] if prompt is None else prompt,
        sampling=SamplingParams(max_new_tokens=4),
        arrival_ns=100,
    )


def test_initial_request_is_pending_prefill() -> None:
    request = make_request()

    assert request.status is RequestStatus.PENDING
    assert request.work_class is WorkClass.PREFILL
    assert request.computed_tokens == 0
    assert request.num_uncomputed_tokens == 3
    assert request.remaining_new_tokens == 4
    assert request.token_slice(0, 2) == [11, 12]


def test_partial_prefill() -> None:
    request = make_request()
    request.status = RequestStatus.RUNNING
    request.active_slot = 0
    request.admitted_ns = 110
    request.computed_tokens = 2

    assert request.work_class is WorkClass.PREFILL
    assert request.num_uncomputed_tokens == 1


def test_steady_decode() -> None:
    request = make_request()
    request.status = RequestStatus.RUNNING
    request.active_slot = 0
    request.admitted_ns = 110
    request.computed_tokens = 3
    request.output_token_ids.append(42)
    request.first_token_ns = 120
    request.work_class = WorkClass.DECODE

    assert request.computed_tokens + 1 == request.total_tokens
    assert request.token_slice(3, 4) == [42]


def test_recompute_preemption() -> None:
    request = make_request()
    request.status = RequestStatus.WAITING
    request.admitted_ns = 110
    request.output_token_ids.append(42)
    request.first_token_ns = 120
    request.preemption_count = 1

    assert request.computed_tokens == 0
    assert request.work_class is WorkClass.PREFILL
    assert request.context_token_ids == [11, 12, 13, 42]


def test_future_request_can_be_aborted_before_arrival() -> None:
    request = RequestState(
        request_id=1,
        submit_seq=1,
        prompt_token_ids=[11],
        sampling=SamplingParams(),
        arrival_ns=100,
        status=RequestStatus.ABORTED,
        finished_ns=50,
    )

    assert request.terminal


@pytest.mark.parametrize("prompt", [[], [-1]])
def test_request_rejects_invalid_prompt(prompt: list[int]) -> None:
    with pytest.raises(ValueError):
        make_request(prompt)
