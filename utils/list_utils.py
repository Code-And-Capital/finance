from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def normalize_to_list(value: Any) -> list[Any] | None:
    """Normalize a scalar/iterable to list, preserving None."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    raise TypeError("Input must be a string or a sequence of strings")


def normalize_string_list(
    values: Any,
    *,
    field_name: str = "values",
) -> list[str] | None:
    """Normalize and validate a list-like input to non-empty strings."""
    normalized = normalize_to_list(values)
    if normalized is None:
        return None

    cleaned: list[str] = []
    for item in normalized:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} must contain only strings")
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"{field_name} must not contain empty strings")
        cleaned.append(stripped)

    return cleaned
