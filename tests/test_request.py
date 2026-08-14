import pytest

from tinyvllm import RequestStatus, SamplingParams, WorkClass


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


def test_request_enums() -> None:
    assert RequestStatus.FINISHED.terminal
    assert RequestStatus.ABORTED.terminal
    assert not RequestStatus.RUNNING.terminal
    assert WorkClass.PREFILL is not WorkClass.DECODE