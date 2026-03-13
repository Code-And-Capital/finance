import abc
from typing import Any, Iterable

import numpy as np
import pandas as pd

from bt.algos.core import Algo


class Factor(Algo, metaclass=abc.ABCMeta):
    """Base class for factor algos with standardized cross-sectional stats."""

    STATS_COLUMNS = ["TOTAL_COVERED", "MEAN", "MEDIAN", "25TH", "75TH"]

    def __init__(self, factor_key: str, name: str | None = None) -> None:
        super().__init__(name=name)
        if not isinstance(factor_key, str) or not factor_key:
            raise TypeError("Factor `factor_key` must be a non-empty string.")
        self.factor_key = factor_key
        self.stats = pd.DataFrame(columns=self.STATS_COLUMNS)

    @staticmethod
    def _to_factor_series(factor_values: Any) -> pd.Series:
        """Normalize factor output to a pandas Series when possible."""
        if isinstance(factor_values, pd.Series):
            return factor_values
        if isinstance(factor_values, pd.DataFrame) and len(factor_values.index) == 1:
            row = factor_values.iloc[0]
            if isinstance(row, pd.Series):
                return row
        return pd.Series(dtype=float)

    def _update_stats(
        self,
        now: pd.Timestamp,
        factor_values: Any,
        investable_universe: Iterable[Any],
    ) -> None:
        """Record summary stats for the factor over the investable universe."""
        factor_series = self._to_factor_series(factor_values)
        investable_index = pd.Index(list(investable_universe))
        factor_series = factor_series.reindex(investable_index)

        numeric = pd.to_numeric(factor_series, errors="coerce")
        finite_mask = numeric.notna() & np.isfinite(numeric.astype(float))
        covered = numeric.loc[finite_mask].astype(float)

        if covered.empty:
            row = [0, np.nan, np.nan, np.nan, np.nan]
        else:
            row = [
                int(len(covered)),
                float(covered.mean()),
                float(covered.median()),
                float(covered.quantile(0.25)),
                float(covered.quantile(0.75)),
            ]

        self.stats.loc[now, self.STATS_COLUMNS] = row

    def __call__(self, target: Any) -> bool:
        """Compute factor for active names and persist it in ``target.temp``."""
        context = self._resolve_temp_universe_now(target)
        if context is None:
            return False
        temp, universe, now = context

        selected = self._resolve_candidate_pool_with_fallback(
            temp,
            lambda: temp.__setitem__("selected", list(universe.columns)) or True,
            allowed_candidates=list(universe.columns),
        )
        if selected is None:
            return False
        if not selected:
            factor_values = pd.Series(dtype=float)
            temp[self.factor_key] = factor_values
            self._update_stats(now, factor_values, selected)
            return True

        factor_values = self.calculate_factor(
            temp=temp,
            universe=universe,
            now=now,
            selected=selected,
        )
        if not isinstance(factor_values, pd.Series):
            return False

        temp[self.factor_key] = factor_values
        self._update_stats(now, factor_values, selected)
        return True

    @abc.abstractmethod
    def calculate_factor(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
    ) -> pd.Series | None:
        """Return a factor Series indexed by security name."""
