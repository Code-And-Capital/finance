"""Classic expected-return estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .core import ExpectedReturns


class SimpleReturn(ExpectedReturns):
    """Estimate expected returns with arithmetic mean historical returns.

    The estimator computes:

    ``E[r_i] = mean_t(r_{t,i})``

    over the return-history window constructed by :class:`ExpectedReturns`.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Historical lookback window used to build returns history.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)

    def calculate_expected_returns(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.Series | None:
        """Return arithmetic mean of historical returns per selected asset."""
        return returns_history[selected].mean(axis=0)


class LogReturn(ExpectedReturns):
    """Estimate expected returns with mean log returns.

    The estimator computes:

    ``E[r_i] = mean_t(log(1 + r_{t,i}))``

    over the return-history window constructed by :class:`ExpectedReturns`.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Historical lookback window used to build returns history.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)

    def calculate_expected_returns(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.Series | None:
        """Return mean log-return per selected asset."""
        return pd.DataFrame(
            np.log1p(returns_history[selected]),
            index=returns_history.index,
            columns=selected,
        ).mean(axis=0)


class MedianReturn(ExpectedReturns):
    """Estimate expected returns with median historical returns.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Historical lookback window used to build returns history.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)

    def calculate_expected_returns(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.Series | None:
        """Return median of historical returns per selected asset."""
        return returns_history[selected].median(axis=0)
