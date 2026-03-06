from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from utils.date_utils import coerce_timestamp_or_none
from utils.list_utils import keep_items_in_pool, normalize_to_list


class Algo:
    """Base class for all composable strategy algos.

    An algo is a callable unit that receives a target object (typically a
    strategy) and returns ``True`` or ``False`` to indicate success / whether a
    condition was met. This class provides common helper methods for:
    - target context resolution (``temp``, ``universe``, ``now``)
    - wide-data row resolution at a timestamp
    - candidate-pool resolution via ``temp['selected']`` with fallback behavior
    - lightweight ticker filtering by current-row availability/sign
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = name
        self.run_always = False

    @property
    def name(self) -> str:
        """Algo name, defaulting to class name when not explicitly set."""
        if self._name is None:
            self._name = self.__class__.__name__
        return self._name

    def __call__(self, target: Any) -> bool:
        """Execute algo logic on ``target``.

        Subclasses must implement this method.
        """
        raise NotImplementedError(f"{self.name} not implemented!")

    def _resolve_temp(self, target: Any) -> dict[str, Any] | None:
        """Return ``target.temp`` when available and dict-like."""
        try:
            temp = getattr(target, "temp", None)
        except Exception:
            return None
        if not isinstance(temp, dict):
            return None
        return temp

    def _resolve_now(self, target: Any) -> pd.Timestamp | None:
        """Return normalized ``target.now`` timestamp when valid."""
        try:
            raw_now = getattr(target, "now", None)
        except Exception:
            return None
        return coerce_timestamp_or_none(raw_now)

    def _resolve_universe(self, target: Any) -> pd.DataFrame | None:
        """Return ``target.universe`` when available and DataFrame-like."""
        try:
            universe = getattr(target, "universe", None)
        except Exception:
            return None
        if not isinstance(universe, pd.DataFrame):
            return None
        return universe

    def _resolve_now_in_universe_or_none(
        self, target: Any, universe: pd.DataFrame
    ) -> pd.Timestamp | None:
        """Return normalized ``target.now`` only when present in universe index."""
        now = self._resolve_now(target)
        if now is None:
            return None
        if now not in universe.index:
            return None
        return now

    def _resolve_selection_context(
        self,
        target: Any,
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.Timestamp] | None:
        """Resolve ``(temp, universe, now)`` for typical algo execution."""
        temp = self._resolve_temp(target)
        universe = self._resolve_universe(target)
        if temp is None or universe is None:
            return None
        now = self._resolve_now_in_universe_or_none(target, universe)
        if now is None:
            return None
        return temp, universe, now

    def _resolve_temp_universe_now(
        self,
        target: Any,
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.Timestamp] | None:
        """Resolve common ``temp``/``universe``/``now`` context."""
        return self._resolve_selection_context(target)

    def _resolve_wide_data_row_at_now(
        self,
        now: pd.Timestamp,
        inline_wide: pd.DataFrame | None,
        wide_key: str | None,
        key_resolver: Callable[[str], Any],
    ) -> tuple[pd.DataFrame, pd.Series] | None:
        """Resolve wide-data source and return the row at ``now``.

        Returns ``None`` when source resolution fails, ``now`` is not present
        in the index, or the resolved row is not a Series.
        """
        wide_df = self._resolve_wide_data_frame(
            inline_wide=inline_wide,
            wide_key=wide_key,
            key_resolver=key_resolver,
        )
        if wide_df is None:
            return None
        if now not in wide_df.index:
            return None
        row = wide_df.loc[now]
        if not isinstance(row, pd.Series):
            return None
        return wide_df, row

    def _resolve_wide_data_frame(
        self,
        inline_wide: pd.DataFrame | None,
        wide_key: str | None,
        key_resolver: Callable[[str], Any],
    ) -> pd.DataFrame | None:
        """Resolve a wide DataFrame from inline data or ``key_resolver``."""
        try:
            wide_df = inline_wide if wide_key is None else key_resolver(wide_key)
        except Exception:
            return None
        if not isinstance(wide_df, pd.DataFrame):
            return None
        return wide_df

    def _signal_row_to_bool_mask(self, row: pd.Series) -> pd.Series:
        """Convert a signal row to a robust boolean mask."""
        try:
            return row.fillna(False).astype(bool)
        except (TypeError, ValueError):
            return pd.Series(False, index=row.index)

    def _filter_tickers_by_current_price(
        self,
        universe: pd.DataFrame,
        now: pd.Timestamp,
        tickers: list[str],
        include_no_data: bool,
        include_negative: bool,
    ) -> list[str]:
        """Filter candidate tickers by current-row availability/sign constraints.

        Parameters
        ----------
        universe
            Wide price DataFrame with timestamp index and ticker columns.
        now
            Evaluation timestamp expected in ``universe.index``.
        tickers
            Candidate tickers to evaluate.
        include_no_data
            If ``True``, return ``tickers`` unchanged.
        include_negative
            If ``True``, keep all non-null prices (including non-positive);
            otherwise keep only strictly positive prices.
        """
        if include_no_data:
            return tickers

        try:
            row = universe.loc[now, tickers].dropna()
        except (TypeError, KeyError, ValueError):
            return []
        if include_negative:
            return list(row.index)
        return list(row[row > 0].index)

    def _resolve_candidate_pool_with_fallback(
        self,
        temp: dict[str, Any],
        fallback_selector: Callable[[], bool],
        allowed_candidates: list[Any] | None = None,
    ) -> list[Any] | None:
        """Return resolved candidate pool from ``temp['selected']``.

        Resolution rules:
        1. Read ``temp['selected']`` and normalize to list.
        2. If missing/empty, run ``fallback_selector`` and retry.
        3. Optionally intersect with ``allowed_candidates`` preserving order.

        Returns ``None`` when normalization fails or fallback fails.
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
        return keep_items_in_pool(allowed_candidates, candidate_pool)
