from __future__ import annotations


def _check_non_negative_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _check_positive_int(name: str, value: int) -> None:
    _check_non_negative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


def _check_token_id(token_id: int) -> None:
    _check_non_negative_int("token_id", token_id)


def _require_model_type(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("model_type must be a non-empty string")
    return value