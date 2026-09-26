from __future__ import annotations
from collections.abc import Iterable


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


def _normalize_token_ids(
    name: str,
    values: int | Iterable[int] | None,
) -> frozenset[int]:
    if values is None:
        return frozenset()

    if type(values) is int:
        values = (values,)
    elif isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must contain integer token IDs")

    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an integer or an iterable of integers"
        ) from exc

    result: set[int] = set()

    for token_id in iterator:
        if type(token_id) is not int:
            raise TypeError(f"{name} must contain interget token IDs")

        if token_id < 0:
            raise ValueError(
                f"{name} must contain non-negative token IDs"
            )

        result.add(token_id)

    return frozenset(result)