from collections.abc import Callable
from typing import Any

import pandas as pd

from utils.date_utils import coerce_timestamp_or_none
from utils.list_utils import normalize_to_list


def resolve_now_in_universe_or_none(
    target: Any, universe: pd.DataFrame
) -> pd.Timestamp | None:
    """Return normalized ``target.now`` only when present in ``universe.index``."""
    try:
        raw_now = getattr(target, "now", None)
    except Exception:
        return None

    now = coerce_timestamp_or_none(raw_now)
    if now is None:
        return None
    if now not in universe.index:
        return None
    return now


def resolve_selection_context(
    target: Any,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Timestamp] | None:
    """Resolve ``temp``, ``universe``, and ``now`` for selection algos."""
    try:
        temp = getattr(target, "temp", None)
        universe = getattr(target, "universe", None)
    except Exception:
        return None

    if not isinstance(temp, dict) or not isinstance(universe, pd.DataFrame):
        return None

    now = resolve_now_in_universe_or_none(target, universe)
    if now is None:
        return None
    return temp, universe, now


def filter_tickers_by_current_price(
    universe: pd.DataFrame,
    now: pd.Timestamp,
    tickers: list[str],
    include_no_data: bool,
    include_negative: bool,
) -> list[str]:
    """Filter candidate tickers by current-row availability and sign constraints."""
    if include_no_data:
        return tickers

    row = universe.loc[now, tickers].dropna()
    if include_negative:
        return list(row.index)
    return list(row[row > 0].index)


def resolve_candidate_pool_with_fallback(
    temp: dict[str, Any],
    fallback_selector: Callable[[], bool],
    allowed_candidates: list[Any] | None = None,
) -> list[Any] | None:
    """Return ``temp['selected']`` or populate it using fallback selector.

    Parameters
    ----------
    temp : dict[str, Any]
        Temp-state dictionary expected to optionally contain ``selected``.
    fallback_selector : Callable[[], bool]
        Callable used to populate ``temp['selected']`` when missing/empty.
    allowed_candidates : list[Any] | None, optional
        If provided, return only items present in ``allowed_candidates``, in
        ``allowed_candidates`` order.

    Returns
    -------
    list[Any] | None
        Selected candidate pool as list, or ``None`` when fallback fails or
        ``temp['selected']`` is malformed.
    """
    try:
        candidate_pool = normalize_to_list(temp.get("selected"))
    except TypeError:
        return None
    if not candidate_pool:
        if not fallback_selector():
            return None
        try:
            candidate_pool = normalize_to_list(temp.get("selected", []))
        except TypeError:
            return None
    if candidate_pool is None:
        return []
    if allowed_candidates is None:
        return candidate_pool
    return intersect_candidates_with_pool(allowed_candidates, candidate_pool)


def intersect_candidates_with_pool(
    candidates: list[str], candidate_pool: list[str]
) -> list[str]:
    """Return candidates that are in candidate_pool, preserving candidates order."""
    pool_set = set(candidate_pool)
    return [candidate for candidate in candidates if candidate in pool_set]


def signal_row_to_bool_mask(row: pd.Series) -> pd.Series:
    """Convert a signal row to a robust boolean mask."""
    try:
        return row.fillna(False).astype(bool)
    except (TypeError, ValueError):
        return pd.Series(False, index=row.index)
