"""Grouped analytics across multiple price series."""

from __future__ import annotations

from typing import Iterable, Union

import pandas as pd

from .performance import TimeSeriesPerformanceStats


class MultiSeriesPerformanceStats(dict[str, TimeSeriesPerformanceStats]):
    """Compute performance analytics for multiple price series.

    Parameters
    ----------
    *prices : pandas.Series | pandas.DataFrame
        One or more Series/DataFrames containing price series. DataFrames are
        expanded by columns. Series names are used as keys.
    rf : float | pandas.Series, optional
        Risk-free input forwarded to each ``TimeSeriesPerformanceStats``.
        annualization_factor : int, optional
        Annualization factor forwarded to each ``TimeSeriesPerformanceStats``.
    """

    def __init__(
        self,
        *prices: Union[pd.Series, pd.DataFrame],
        rf: Union[float, pd.Series] = 0.0,
        annualization_factor: int = 252,
    ) -> None:
        super().__init__()
        normalized = self._normalize_inputs(prices)
        if normalized.empty:
            raise ValueError(
                "MultiSeriesPerformanceStats requires at least one price series."
            )

        self._names = list(normalized.columns)
        self.prices = normalized

        for name in self._names:
            self[name] = TimeSeriesPerformanceStats(
                self.prices[name],
                rf=rf,
                annualization_factor=annualization_factor,
            )

        self.stats = pd.DataFrame({name: self[name].stats for name in self._names})

    @staticmethod
    def _normalize_inputs(
        values: Iterable[Union[pd.Series, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Normalize mixed Series/DataFrame inputs to one wide DataFrame."""
        series_list: list[pd.Series] = []
        seen_names: set[str] = set()
        unnamed_count = 0

        for value in values:
            if isinstance(value, pd.DataFrame):
                for col in value.columns:
                    key = str(col)
                    if key in seen_names:
                        raise ValueError(f"Duplicate series name '{key}' in inputs.")
                    seen_names.add(key)
                    series_list.append(value[col].rename(key))
                continue

            if isinstance(value, pd.Series):
                if value.name is None:
                    key = f"series_{unnamed_count}"
                    unnamed_count += 1
                else:
                    key = str(value.name)
                if key in seen_names:
                    raise ValueError(f"Duplicate series name '{key}' in inputs.")
                seen_names.add(key)
                series_list.append(value.rename(key))
                continue

            raise TypeError(
                "MultiSeriesPerformanceStats inputs must be pandas Series or DataFrame."
            )

        if not series_list:
            return pd.DataFrame()
        return pd.concat(series_list, axis=1)

    def __getitem__(self, key: int | str) -> TimeSeriesPerformanceStats:
        """Return member analytics by integer position or string name."""
        if isinstance(key, int):
            if key < 0 or key >= len(self._names):
                raise IndexError(
                    f"Group index {key} out of bounds for {len(self._names)} series."
                )
            return super().__getitem__(self._names[key])
        return super().__getitem__(key)
