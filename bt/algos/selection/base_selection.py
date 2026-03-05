from collections.abc import Callable
from typing import Any

import pandas as pd

from bt.algos.core import Algo
from utils.list_utils import keep_items_in_pool
from utils.math_utils import is_zero


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
        context = self._resolve_selection_context(target)
        if context is None:
            return False
        temp, universe, now = context

        candidates = list(universe.columns)
        temp["selected"] = self._filter_tickers_by_current_price(
            universe=universe,
            now=now,
            tickers=candidates,
            include_no_data=self.include_no_data,
            include_negative=self.include_negative,
        )
        return True

    def _resolve_context_and_candidate_pool(
        self,
        target: Any,
        fallback_selector: Callable[[], bool],
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.Timestamp, list[Any]] | None:
        """Resolve selection context and candidate pool in one shared step."""
        context = self._resolve_selection_context(target)
        if context is None:
            return None
        temp, universe, now = context

        candidate_pool = self._resolve_candidate_pool_with_fallback(
            temp,
            fallback_selector,
            allowed_candidates=list(universe.columns),
        )
        if candidate_pool is None:
            return None
        return temp, universe, now, candidate_pool


class SelectHasData(SelectAll):
    """Select names with complete non-null history over a lookback window.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Historical lookback window ending at ``target.now``.

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
    ) -> None:
        """Initialize data-availability selector."""
        super().__init__()
        if not isinstance(lookback, pd.DateOffset):
            raise TypeError("SelectHasData `lookback` must be a pandas.DateOffset.")
        self.lookback = lookback

    def __call__(self, target: Any) -> bool:
        """Filter selection by historical data availability and current prices."""
        resolved = self._resolve_context_and_candidate_pool(
            target,
            lambda: super(SelectHasData, self).__call__(target),
        )
        if resolved is None:
            return False
        temp, universe, now, candidate_pool = resolved
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
        temp["selected"] = self._filter_tickers_by_current_price(
            universe=universe,
            now=now,
            tickers=candidates,
            include_no_data=False,
            include_negative=False,
        )
        return True


class SelectActive(SelectAll):
    """Drop inactive names from ``target.temp['selected']``.

    Notes
    -----
    - Reads ``target.temp['selected']`` as the candidate list.
      If missing or empty, selection is first populated via :class:`SelectAll`.
    - Reads ``target.perm['rolled']`` and ``target.perm['closed']`` as inactive
      names. If missing, empty collections are assumed.
    - Output order matches the original ``selected`` order.
    """

    def __init__(self) -> None:
        """Initialize active-name selector."""
        super().__init__()

    def __call__(self, target: Any) -> bool:
        """Keep active names and store result in ``target.temp['selected']``.

        Returns
        -------
        bool
            ``True`` when state was processed. Returns ``False`` when
            ``target.temp`` or ``target.perm`` is missing or not dict-like.
        """
        perm = getattr(target, "perm", None)
        if not isinstance(perm, dict):
            return False

        resolved = self._resolve_context_and_candidate_pool(
            target,
            lambda: super(SelectActive, self).__call__(target),
        )
        if resolved is None:
            return False
        temp, _, _, candidate_pool = resolved

        rolled = set(perm.get("rolled", set()))
        closed = set(perm.get("closed", set()))
        inactive = rolled.union(closed)
        temp["selected"] = [name for name in candidate_pool if name not in inactive]
        return True


class SelectIsOpen(SelectAll):
    """Keep only currently open positions from the candidate selection list.

    This selector inspects ``target.children`` and marks a child as open when
    its current ``weight`` is non-zero (within numerical tolerance). Open names
    are filtered from the current candidate pool and written to
    ``target.temp['selected']``.

    Notes
    -----
    - Candidate pool source is ``target.temp['selected']``.
      If missing or empty, :class:`SelectAll` is run first.
    - Only child names present in ``target.universe.columns`` are eligible.
    - Output order is deterministic and follows candidate-pool order.
    - Returns ``False`` for malformed state (invalid context/children/weights).
    """

    def __init__(self) -> None:
        """Initialize open-position selector."""
        super().__init__()

    def __call__(self, target: Any) -> bool:
        """Filter ``target.temp['selected']`` to names with open positions."""
        children = getattr(target, "children", None)
        if not isinstance(children, dict):
            return False

        resolved = self._resolve_context_and_candidate_pool(
            target,
            lambda: super(SelectIsOpen, self).__call__(target),
        )
        if resolved is None:
            return False
        temp, _, _, candidate_pool = resolved

        open_candidates: list[str] = []
        for candidate_name in candidate_pool:
            child = children.get(candidate_name)
            if child is None:
                continue

            try:
                child_weight = child.weight
                if not is_zero(child_weight):
                    open_candidates.append(candidate_name)
            except Exception:
                return False

        temp["selected"] = open_candidates
        return True
