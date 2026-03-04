from typing import Any

import pandas as pd
import numpy as np

from bt.core.algo_base import Algo
from bt.utils.selection_utils import resolve_selection_context
from utils.math_utils import validate_integer


class SelectN(Algo):
    """Select top/bottom names by rank metric in ``target.temp[stat_key]``.

    Parameters
    ----------
    n : int | float
        Selection size. ``0 < n < 1`` means percentage, ``n >= 1`` means
        absolute count.
    sort_descending : bool, optional
        If ``True`` (default), highest metric values rank first.
    stat_key : str, optional
        Key in ``target.temp`` containing the ranking ``pandas.Series``.
        Defaults to ``"stat"``.

    Notes
    -----
    - For percentage selection, count is computed as
      ``max(floor(n * len(ranked)), 1)``.
    - Returns ``False`` (without mutating selection) when ``target.temp`` is
      missing/not dict-like, when ``temp[stat_key]`` is missing/non-Series, or
      when ``temp['selected']`` is missing.
    - Ranking is always limited to names already present in ``temp['selected']``.
    """

    def __init__(
        self,
        n: float | int,
        sort_descending: bool = True,
        stat_key: str = "stat",
    ) -> None:
        """Initialize rank selector.

        Raises
        ------
        TypeError
            If ``n`` is not numeric, if absolute ``n`` is non-integer, or when
            ``sort_descending`` is not ``bool``.
        ValueError
            If ``n <= 0``.
        """
        super().__init__()
        if not isinstance(sort_descending, bool):
            raise TypeError("SelectN `sort_descending` must be a bool.")
        if not isinstance(stat_key, str) or not stat_key:
            raise TypeError("SelectN `stat_key` must be a non-empty string.")

        if isinstance(n, bool) or not isinstance(n, (int, float)):
            raise TypeError("SelectN `n` must be numeric.")
        if n <= 0:
            raise ValueError("SelectN `n` must be > 0.")
        if n >= 1:
            # Absolute-count mode is strict integer to avoid silent truncation.
            n = int(validate_integer(n, "SelectN `n`"))

        self.n = n
        self.ascending = not sort_descending
        self.stat_key = stat_key

    def __call__(self, target: Any) -> bool:
        """Rank and select names, storing output in ``target.temp['selected']``."""
        context = resolve_selection_context(target)
        if context is None:
            return False
        temp, _, _ = context

        stat = temp.get(self.stat_key)
        if not isinstance(stat, pd.Series):
            return False

        if "selected" not in temp:
            return False

        ranked = stat.dropna()
        ranked = ranked[np.isfinite(ranked)]
        ranked = ranked.loc[ranked.index.intersection(temp["selected"])]

        if ranked.empty:
            temp["selected"] = []
            return True

        ranked = ranked.sort_values(ascending=self.ascending)
        if 0 < self.n < 1:
            keep_n = max(int(self.n * len(ranked)), 1)
        else:
            keep_n = int(self.n)

        top_names = list(ranked.iloc[:keep_n].index)
        temp["selected"] = top_names
        return True
