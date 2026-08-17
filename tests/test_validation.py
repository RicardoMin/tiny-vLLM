from __future__ import annotations

import pytest

from tinyvllm._validation import (
    _check_non_negative_int,
    _check_optional_non_negative_int,
    _check_positive_int,
    _check_token_id,
)


def test_non_negative_integer_contract() -> None:
    _check_non_negative_int("value", 0)
    _check_non_negative_int("value", 3)

    with pytest.raises(TypeError, match="integer"):
        _check_non_negative_int("value", True)
    with pytest.raises(ValueError, match="non-negative"):
        _check_non_negative_int("value", -1)


def test_positive_integer_contract() -> None:
    _check_positive_int("value", 1)

    with pytest.raises(ValueError, match="positive"):
        _check_positive_int("value", 0)


def test_token_id_uses_non_negative_contract() -> None:
    _check_token_id(0)

    with pytest.raises(ValueError, match="non-negative"):
        _check_token_id(-1)


def test_optional_integer_contract() -> None:
    _check_optional_non_negative_int("value", None)
    _check_optional_non_negative_int("value", 0)

    with pytest.raises(ValueError, match="non-negative"):
        _check_optional_non_negative_int("value", -1)
