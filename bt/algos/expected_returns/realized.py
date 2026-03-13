"""Forward-window expected-return estimators."""

from typing import Any

import pandas as pd

from .core import ExpectedReturns


class RealizedReturn(ExpectedReturns):
    """Estimate expected returns over a lookback + lookforward window.

    This estimator extends the base return-history window by including
    ``lookforward`` past ``now - lag`` and delegates expected-return
    computation to another :class:`ExpectedReturns` estimator.

    Parameters
    ----------
    expected_return_estimator : ExpectedReturns
        Estimator used to compute expected returns from realized-window
        returns history.
    lookback : pandas.DateOffset, optional
        Historical lookback window used to build returns history.
    lookforward : pandas.DateOffset, optional
        Forward extension added to the estimation window.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and center of estimation window.
    """

    def __init__(
        self,
        expected_return_estimator: ExpectedReturns,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lookforward: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        if not isinstance(expected_return_estimator, ExpectedReturns):
            raise TypeError(
                "RealizedReturn `expected_return_estimator` must be an ExpectedReturns instance."
            )
        if not isinstance(lookforward, pd.DateOffset):
            raise TypeError("RealizedReturn `lookforward` must be a pandas.DateOffset.")
        self.expected_return_estimator = expected_return_estimator
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
