from __future__ import annotations

from typing import Any

import pandas as pd


def coerce_timestamp(value: Any, label: str = "value") -> pd.Timestamp:
    """
    Convert a date-like value to a valid ``pandas.Timestamp``.

    Parameters
    ----------
    value : Any
        Date-like input accepted by :func:`pandas.to_datetime`.
    label : str, optional
        Field label used in error messages.

    Returns
    -------
    pandas.Timestamp
        Parsed timestamp.

    Raises
    ------
    ValueError
        If the input cannot be parsed into a valid timestamp.
    """
    try:
        ts = pd.to_datetime(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid timestamp.") from exc

    if pd.isna(ts):
        raise ValueError(f"{label} must be a valid timestamp.")
    return pd.Timestamp(ts)


def coerce_timestamp_or_none(value: Any) -> pd.Timestamp | None:
    """
    Convert a date-like value to ``pandas.Timestamp`` or return ``None``.

    Parameters
    ----------
    value : Any
        Date-like input.

    Returns
    -------
    pandas.Timestamp | None
        Parsed timestamp, or ``None`` when input is missing/invalid.
    """
    if value is None:
        return None

    try:
        return coerce_timestamp(value)
    except ValueError:
        return None


def month_index(value: Any, label: str = "value") -> int:
    """
    Convert a date-like value to its calendar month index.

    Parameters
    ----------
    value : Any
        Date-like input accepted by :func:`pandas.to_datetime`.
    label : str, optional
        Field label used in error messages when parsing fails.

    Returns
    -------
    int
        Calendar month index computed as ``year * 12 + month``.
    """
    ts = coerce_timestamp(value, label=label)
    return ts.year * 12 + ts.month
