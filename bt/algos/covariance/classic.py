"""Historical covariance estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bt.algos.covariance.core import Covariance


class SimpleCovariance(Covariance):
    """Sample covariance estimator on arithmetic returns.

    This estimator applies ``DataFrame.cov`` directly to arithmetic returns
    prepared by :class:`Covariance`.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    ddof : int, optional
        Delta degrees of freedom passed to ``DataFrame.cov``.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
        ddof: int = 1,
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        if not isinstance(ddof, int):
            raise TypeError("SimpleCovariance `ddof` must be an integer.")
        if ddof < 0:
            raise ValueError("SimpleCovariance `ddof` must be >= 0.")
        self.ddof = ddof

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return covariance matrix from arithmetic returns.

        Parameters
        ----------
        temp : dict[str, Any]
            Strategy temporary storage dictionary.
        universe : pandas.DataFrame
            Price universe.
        now : pandas.Timestamp
            Evaluation timestamp.
        selected : list[Any]
            Selected asset names.
        returns_history : pandas.DataFrame
            Arithmetic return matrix from base class preprocessing.
        """
        selected_returns = returns_history.reindex(columns=selected)
        if selected_returns.empty:
            return pd.DataFrame(index=selected, columns=selected, dtype=float)
        return selected_returns.cov(ddof=self.ddof)


class LogCovariance(Covariance):
    """Sample covariance estimator on log returns.

    The estimator transforms arithmetic returns using ``np.log1p`` and then
    applies ``DataFrame.cov``.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    ddof : int, optional
        Delta degrees of freedom passed to ``DataFrame.cov``.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
        ddof: int = 1,
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        if not isinstance(ddof, int):
            raise TypeError("LogCovariance `ddof` must be an integer.")
        if ddof < 0:
            raise ValueError("LogCovariance `ddof` must be >= 0.")
        self.ddof = ddof

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return covariance matrix from ``log(1 + returns)``.

        Parameters
        ----------
        temp : dict[str, Any]
            Strategy temporary storage dictionary.
        universe : pandas.DataFrame
            Price universe.
        now : pandas.Timestamp
            Evaluation timestamp.
        selected : list[Any]
            Selected asset names.
        returns_history : pandas.DataFrame
            Arithmetic return matrix from base class preprocessing.
        """
        selected_returns = returns_history.reindex(columns=selected)
        if selected_returns.empty:
            return pd.DataFrame(index=selected, columns=selected, dtype=float)
        log_returns = np.log1p(selected_returns)
        return log_returns.cov(ddof=self.ddof)
