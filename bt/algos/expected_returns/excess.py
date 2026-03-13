"""Benchmark-relative expected-return estimators."""

from typing import Any

import pandas as pd

from utils.dataframe_utils import one_column_frame_to_series

from .core import ExpectedReturns


class ExcessReturn(ExpectedReturns):
    """Estimate expected returns from benchmark-relative excess returns.

    This wrapper computes asset excess returns over a benchmark and delegates
    expected-return computation to another :class:`ExpectedReturns` estimator.

    Parameters
    ----------
    index_data_key : str
        Setup-data key used with ``target.get_data(index_data_key)`` to load
        benchmark price history (Series or one-column DataFrame).
    expected_return_estimator : ExpectedReturns
        Estimator used to compute expected returns from excess returns history.
    lookback : pandas.DateOffset, optional
        Historical lookback window used to build returns history.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        index_data_key: str,
        expected_return_estimator: ExpectedReturns,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        if not isinstance(index_data_key, str) or not index_data_key:
            raise TypeError("ExcessReturn `index_data_key` must be a non-empty string.")
        if not isinstance(expected_return_estimator, ExpectedReturns):
            raise TypeError(
                "ExcessReturn `expected_return_estimator` must be an ExpectedReturns instance."
            )
        self.index_data_key = index_data_key
        self.expected_return_estimator = expected_return_estimator

    def _build_returns_history(
        self,
        target: Any,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
    ) -> pd.DataFrame:
        """Build benchmark-relative excess-return history over lookback window."""
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

        try:
            benchmark_prices = benchmark.loc[start:end]
        except Exception:
            return pd.DataFrame()
        if benchmark_prices.empty:
            return pd.DataFrame()

        asset_returns = prices.pct_change().iloc[1:]
        benchmark_returns = benchmark_prices.pct_change().iloc[1:]
        excess = asset_returns.sub(benchmark_returns, axis=0)
        return excess.replace([float("inf"), -float("inf")], pd.NA).astype(float)

    def calculate_expected_returns(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.Series | None:
        """Delegate expected-return computation to wrapped estimator."""
        return self.expected_return_estimator.calculate_expected_returns(
            temp=temp,
            universe=universe,
            now=now,
            selected=selected,
            returns_history=returns_history,
        )
