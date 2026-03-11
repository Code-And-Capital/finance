"""Benchmark-relative covariance estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bt.algos.covariance.core import Covariance
from utils.dataframe_utils import one_column_frame_to_series


class ExcessCovariance(Covariance):
    """Estimate covariance of asset excess returns over a benchmark.

    Parameters
    ----------
    index_data_key : str
        Setup-data key used with ``target.get_data(index_data_key)`` to load
        benchmark price history (Series or one-column DataFrame).
    covariance_estimator : Covariance
        Covariance estimator used to compute covariance from excess returns.
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        index_data_key: str,
        covariance_estimator: Covariance,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        if not isinstance(index_data_key, str) or not index_data_key:
            raise TypeError(
                "ExcessCovariance `index_data_key` must be a non-empty string."
            )
        if not isinstance(covariance_estimator, Covariance):
            raise TypeError(
                "ExcessCovariance `covariance_estimator` must be a Covariance instance."
            )
        self.index_data_key = index_data_key
        self.covariance_estimator = covariance_estimator

    def _build_returns_history(
        self,
        target: Any,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
    ) -> pd.DataFrame:
        """Build excess-return history versus benchmark over lookback window."""
        end = now - self.lag
        start = now - self.lookback

        try:
            prices = universe.loc[start:end, selected]
        except (TypeError, KeyError, ValueError):
            return pd.DataFrame()
        if prices.empty:
            return pd.DataFrame()

        try:
            benchmark_source = target.get_data(self.index_data_key)
        except Exception:
            return pd.DataFrame()
        if isinstance(benchmark_source, pd.Series):
            benchmark = benchmark_source
        elif isinstance(benchmark_source, pd.DataFrame):
            try:
                benchmark = one_column_frame_to_series(benchmark_source)
            except ValueError:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
        if benchmark is None:
            return pd.DataFrame()

        try:
            benchmark_prices = benchmark.loc[start:end]
        except Exception:
            return pd.DataFrame()
        if benchmark_prices.empty:
            return pd.DataFrame()

        asset_returns = prices.pct_change().iloc[1:]
        benchmark_returns = benchmark_prices.pct_change().iloc[1:]
        excess = asset_returns.sub(benchmark_returns, axis=0)
        return excess.replace([np.inf, -np.inf], np.nan)

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Delegate covariance computation to wrapped estimator."""
        return self.covariance_estimator.calculate_covariance(
            temp=temp,
            universe=universe,
            now=now,
            selected=selected,
            returns_history=returns_history,
        )
