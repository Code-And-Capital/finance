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


def keep_items_in_pool(items: list[Any], pool: list[Any]) -> list[Any]:
    """Return items present in pool, preserving item order."""
    pool_set = set(pool)
    return [item for item in items if item in pool_set]


def drop_items_in_pool(items: list[Any], excluded_items: Iterable[Any]) -> list[Any]:
    """Return items not present in excluded_items, preserving item order."""
    excluded_set = set(excluded_items)
    return [item for item in items if item not in excluded_set]
