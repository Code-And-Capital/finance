from collections.abc import Iterable
from typing import Any

import pandas as pd

from bt.utils.selection_utils import (
    filter_tickers_by_current_price,
    intersect_candidates_with_pool,
    resolve_candidate_pool_with_fallback,
    resolve_selection_context,
    signal_row_to_bool_mask,
)
from utils.list_utils import normalize_string_list
from .base_selection import SelectAll


class SelectThese(SelectAll):
    """Select a fixed ticker list, optionally filtered by current prices.

    Parameters
    ----------
    tickers : Iterable[str] | str
        Tickers to keep. A scalar string is treated as a single ticker.
    include_no_data : bool, optional
        If ``True``, keep configured tickers regardless of missing prices.
    include_negative : bool, optional
        If ``False`` (default), exclude non-positive prices.

    Notes
    -----
    - Candidate pool is taken from ``target.temp['selected']``.
      If missing or empty, :class:`SelectAll` is run first.
    - Only configured tickers that are in the active candidate pool can be selected.
    - Returns ``False`` when ``target.temp`` is missing/not dict-like, universe
      is missing/invalid, or ``target.now`` is missing/invalid/not in index.
    """

    def __init__(
        self,
        tickers: Iterable[str] | str,
        include_no_data: bool = False,
        include_negative: bool = False,
    ) -> None:
        """Initialize fixed-list selector."""
        super().__init__(
            include_no_data=include_no_data, include_negative=include_negative
        )
        normalized = normalize_string_list(tickers, field_name="SelectThese `tickers`")
        if normalized is None:
            raise TypeError("SelectThese `tickers` must be a string or iterable.")

        if not normalized:
            raise ValueError("SelectThese `tickers` must not be empty.")

        self.tickers = sorted(set(normalized))

    def __call__(self, target: Any) -> bool:
        """Write fixed-list filtered selection into ``target.temp['selected']``."""
        context = resolve_selection_context(target)
        if context is None:
            return False
        temp, universe, now = context

        candidate_pool = resolve_candidate_pool_with_fallback(
            temp, lambda: super(SelectThese, self).__call__(target)
        )
        if candidate_pool is None:
            return False

        available = intersect_candidates_with_pool(self.tickers, candidate_pool)
        temp["selected"] = filter_tickers_by_current_price(
            universe=universe,
            now=now,
            tickers=available,
            include_no_data=self.include_no_data,
            include_negative=self.include_negative,
        )
        return True


class SelectWhere(SelectAll):
    """Select names where a boolean signal is true at ``target.now``.

    Parameters
    ----------
    signal : str | pandas.DataFrame
        Signal source. If string, retrieved via ``target.get_data(signal)``.
    include_no_data : bool, optional
        If ``False`` (default), exclude names with missing prices at ``target.now``.
    include_negative : bool, optional
        If ``False`` (default), exclude non-positive prices.

    Notes
    -----
    - Candidate pool is taken from ``target.temp['selected']``.
      If missing or empty, :class:`SelectAll` is run first.
    - Signal is applied only to names in the candidate pool.
    - Returns ``False`` when ``target.temp`` is missing/not dict-like, universe
      is missing/invalid, or ``target.now`` is missing/invalid/not in index.
    """

    def __init__(
        self,
        signal: str | pd.DataFrame,
        include_no_data: bool = False,
        include_negative: bool = False,
    ) -> None:
        """Initialize signal-based selector."""
        super().__init__(
            include_no_data=include_no_data, include_negative=include_negative
        )
        if isinstance(signal, pd.DataFrame):
            self.signal = signal
            self.signal_name: str | None = None
        elif isinstance(signal, str):
            self.signal = None
            self.signal_name = signal
        else:
            raise TypeError(
                "SelectWhere `signal` must be a DataFrame or data key string."
            )

    def __call__(self, target: Any) -> bool:
        """Select signaled names and store them in ``target.temp['selected']``."""
        context = resolve_selection_context(target)
        if context is None:
            return False
        temp, universe, now = context

        signal_df = (
            self.signal
            if self.signal_name is None
            else target.get_data(self.signal_name)
        )
        if not isinstance(signal_df, pd.DataFrame):
            raise TypeError("SelectWhere signal must resolve to a pandas DataFrame.")

        if now not in signal_df.index:
            return False

        candidate_pool = resolve_candidate_pool_with_fallback(
            temp, lambda: super(SelectWhere, self).__call__(target)
        )
        if candidate_pool is None:
            return False

        candidate_pool = intersect_candidates_with_pool(
            list(signal_df.columns), candidate_pool
        )
        if not candidate_pool:
            temp["selected"] = []
            return True

        row = signal_df.loc[now, candidate_pool]
        mask = signal_row_to_bool_mask(row)
        selected_names = list(mask[mask].index)

        temp["selected"] = filter_tickers_by_current_price(
            universe=universe,
            now=now,
            tickers=selected_names,
            include_no_data=self.include_no_data,
            include_negative=self.include_negative,
        )
        return True
