from typing import Any

import pandas as pd

from bt.core.algo_base import Algo
from bt.utils.selection_utils import (
    filter_tickers_by_current_price,
    resolve_selection_context,
    resolve_candidate_pool_with_fallback,
)


class SelectAll(Algo):
    """Select all investable securities in the current universe row.

    Parameters
    ----------
    include_no_data : bool, optional
        If ``True``, include all universe columns regardless of current price
        availability.
    include_negative : bool, optional
        If ``False`` (default), exclude non-positive prices. Ignored when
        ``include_no_data`` is ``True``.
    """

    def __init__(
        self, include_no_data: bool = False, include_negative: bool = False
    ) -> None:
        """Initialize the all-universe selector."""
        super().__init__()
        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target: Any) -> bool:
        """Populate ``target.temp['selected']`` with selected tickers."""
        context = resolve_selection_context(target)
        if context is None:
            return False
        temp, universe, now = context

        candidates = list(universe.columns)
        temp["selected"] = filter_tickers_by_current_price(
            universe=universe,
            now=now,
            tickers=candidates,
            include_no_data=self.include_no_data,
            include_negative=self.include_negative,
        )
        return True


class SelectHasData(SelectAll):
    """Select names with complete non-null history over a lookback window.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Historical lookback window ending at ``target.now``.
    include_no_data : bool, optional
        Passed to :class:`SelectAll` fallback and current-price filtering.
    include_negative : bool, optional
        Passed to :class:`SelectAll` fallback and current-price filtering.

    Notes
    -----
    - Reads ``target.temp['selected']`` as candidate names.
      If missing or empty, candidates are first populated via :class:`SelectAll`.
    - Returns ``False`` when ``target.temp`` is not dict-like, ``target.universe``
      is missing/invalid, or ``target.now`` is missing/invalid/not in index.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        include_no_data: bool = False,
        include_negative: bool = False,
    ) -> None:
        """Initialize data-availability selector."""
        super().__init__(
            include_no_data=include_no_data, include_negative=include_negative
        )
        if not isinstance(lookback, pd.DateOffset):
            raise TypeError("SelectHasData `lookback` must be a pandas.DateOffset.")
        self.lookback = lookback

    def __call__(self, target: Any) -> bool:
        """Filter selection by historical data availability and current prices."""
        context = resolve_selection_context(target)
        if context is None:
            return False
        temp, universe, now = context

        candidate_pool = resolve_candidate_pool_with_fallback(
            temp,
            lambda: super(SelectHasData, self).__call__(target),
            allowed_candidates=list(universe.columns),
        )
        if candidate_pool is None:
            return False
        if not candidate_pool:
            temp["selected"] = []
            return True

        start_date = now - self.lookback
        try:
            history = universe.loc[start_date:now, candidate_pool]
        except (TypeError, KeyError, ValueError):
            return False
        if history.empty:
            temp["selected"] = []
            return True

        complete_history = history.notna().all(axis=0)
        candidates = list(complete_history[complete_history].index)
        temp["selected"] = filter_tickers_by_current_price(
            universe=universe,
            now=now,
            tickers=candidates,
            include_no_data=self.include_no_data,
            include_negative=self.include_negative,
        )
        return True


class SelectActive(SelectAll):
    """Drop inactive names from ``target.temp['selected']``.

    Parameters
    ----------
    include_no_data : bool, optional
        Passed to :class:`SelectAll` when a fallback selection is needed.
    include_negative : bool, optional
        Passed to :class:`SelectAll` when a fallback selection is needed.

    Notes
    -----
    - Reads ``target.temp['selected']`` as the candidate list.
      If missing or empty, selection is first populated via :class:`SelectAll`.
    - Reads ``target.perm['rolled']`` and ``target.perm['closed']`` as inactive
      names. If missing, empty collections are assumed.
    - Output order matches the original ``selected`` order.
    """

    def __init__(
        self, include_no_data: bool = False, include_negative: bool = False
    ) -> None:
        """Initialize active-name selector with SelectAll fallback options."""
        super().__init__(
            include_no_data=include_no_data, include_negative=include_negative
        )

    def __call__(self, target: Any) -> bool:
        """Keep active names and store result in ``target.temp['selected']``.

        Returns
        -------
        bool
            ``True`` when state was processed. Returns ``False`` when
            ``target.temp`` or ``target.perm`` is missing or not dict-like.
        """
        context = resolve_selection_context(target)
        if context is None:
            return False
        temp, universe, _ = context

        perm = getattr(target, "perm", None)
        if not isinstance(perm, dict):
            return False

        candidate_pool = resolve_candidate_pool_with_fallback(
            temp,
            lambda: super(SelectActive, self).__call__(target),
            allowed_candidates=list(universe.columns),
        )
        if candidate_pool is None:
            return False

        rolled = set(perm.get("rolled", set()))
        closed = set(perm.get("closed", set()))
        inactive = rolled.union(closed)
        target.temp["selected"] = [
            name for name in candidate_pool if name not in inactive
        ]
        return True
