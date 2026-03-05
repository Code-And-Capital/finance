from typing import Any

from bt.core.algo_base import Algo
from bt.utils.selection_utils import (
    exclude_candidates_from_pool,
    filter_tickers_by_current_price,
    intersect_candidates_with_pool,
    resolve_candidate_pool_with_fallback,
    resolve_selection_context,
)
from utils.list_utils import normalize_string_list


class RemoveSecurities(Algo):
    """Remove configured securities from the current selection.

    Parameters
    ----------
    removed_securities : list[str] | tuple[str, ...] | str
        Security names that should be removed when present in
        ``target.temp['selected']``.

    Notes
    -----
    - If ``target.temp['selected']`` is missing, output is an empty list.
    - Output order follows the original candidate pool order.
    - Returns ``False`` when ``target.temp`` is missing/not dict-like or when
      configured/loaded security lists are malformed.
    """

    def __init__(self, removed_securities: list[str] | tuple[str, ...] | str) -> None:
        """Initialize removal selector."""
        super().__init__()
        normalized = normalize_string_list(
            removed_securities, field_name="RemoveSecurities `removed_securities`"
        )
        if normalized is None:
            raise TypeError(
                "RemoveSecurities `removed_securities` must be a string or iterable."
            )
        self.removed_securities = set(normalized)

    def __call__(self, target: Any) -> bool:
        """Remove configured names from ``target.temp['selected']``."""
        context = resolve_selection_context(target)
        if context is None:
            return False
        temp, _, _ = context

        candidate_pool = resolve_candidate_pool_with_fallback(
            temp,
            lambda: temp.__setitem__("selected", []) or True,
        )
        if candidate_pool is None:
            return False

        temp["selected"] = exclude_candidates_from_pool(
            candidate_pool, self.removed_securities
        )
        return True


class AddSecurity(Algo):
    """Add configured securities to the current selection.

    Parameters
    ----------
    tickers : list[str] | tuple[str, ...] | str
        Security names to add to the current candidate pool.
    include_negative : bool, optional
        If ``True``, allow zero/negative prices at ``target.now``.
        If ``False`` (default), only strictly positive current prices are kept.

    Notes
    -----
    - Starts from ``target.temp['selected']`` when present, otherwise empty pool.
    - Names listed in ``target.perm['closed']`` are permanently removed from the
      internal ticker universe and will not be added again in later calls.
      This class is intentionally stateful across calls for that cache behavior.
    - Keeps only names present in ``target.universe.columns``.
    - Excludes missing current prices.
    - Returns ``False`` when context/state is malformed.
    """

    def __init__(
        self,
        tickers: list[str] | tuple[str, ...] | str,
        include_negative: bool = False,
    ) -> None:
        """Initialize add-security selector."""
        super().__init__()
        normalized = normalize_string_list(tickers, field_name="AddSecurity `tickers`")
        if normalized is None:
            raise TypeError("AddSecurity `tickers` must be a string or iterable.")
        self.tickers = set(normalized)
        self.include_negative = include_negative

    def __call__(self, target: Any) -> bool:
        """Add configured names and filter to currently tradable securities."""
        context = resolve_selection_context(target)
        if context is None:
            return False
        temp, universe, now = context

        perm = getattr(target, "perm", {})
        if not isinstance(perm, dict):
            return False

        candidate_pool = resolve_candidate_pool_with_fallback(
            temp,
            lambda: temp.__setitem__("selected", []) or True,
        )
        if candidate_pool is None:
            return False

        try:
            closed_set = set(perm.get("closed", []))
        except TypeError:
            return False
        self.tickers.difference_update(closed_set)

        try:
            merged_set = set(candidate_pool).union(self.tickers)
        except TypeError:
            return False

        universe_columns = list(universe.columns)
        universe_candidates = intersect_candidates_with_pool(
            universe_columns, list(merged_set)
        )
        temp["selected"] = filter_tickers_by_current_price(
            universe=universe,
            now=now,
            tickers=universe_candidates,
            include_no_data=False,
            include_negative=self.include_negative,
        )
        return True
