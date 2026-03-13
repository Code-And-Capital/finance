"""Forward-window covariance estimator wrappers."""

from typing import Any

import pandas as pd

from .core import Covariance


class RealizedCovariance(Covariance):
    """Forward-window covariance wrapper around an estimator.

    This class uses the same execution workflow as :class:`Covariance`, but
    changes the return-history window to include a forward segment and delegates
    covariance computation to another estimator instance.

    Parameters
    ----------
    covariance_estimator : Covariance
        Covariance estimator instance used to compute the covariance matrix from
        the realized return history.
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lookforward : pandas.DateOffset, optional
        Forward extension added to the estimation window.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and center of estimation window.
    """

    def __init__(
        self,
        covariance_estimator: Covariance,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lookforward: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        if not isinstance(covariance_estimator, Covariance):
            raise TypeError(
                "RealizedCovariance `covariance_estimator` must be a Covariance instance."
            )
        if not isinstance(lookforward, pd.DateOffset):
            raise TypeError(
                "RealizedCovariance `lookforward` must be a pandas.DateOffset."
            )
        self.covariance_estimator = covariance_estimator
        self.lookforward = lookforward

    def _build_returns_history(
        self,
        target: Any,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
    ) -> pd.DataFrame:
        """Build return history using lookback + lookforward window."""
        eval_time = now - self.lag
        start = now - self.lookback
        end = eval_time + self.lookforward
        return self._returns_from_window(
            universe=universe,
            selected=selected,
            start=start,
            end=end,
        )

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
