from typing import Any

import pandas as pd

from bt.algos.factors.core.factor import Factor


class TotalReturn(Factor):
    """Compute trailing total return and store it in ``target.temp['total_return']``.

    Total return is computed over:
    ``[evaluation_timestamp - lookback, evaluation_timestamp - lag]``.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Lookback window for return computation.
    lag : pandas.DateOffset, optional
        Lag applied to the evaluation timestamp to avoid look-ahead.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
        standardize: bool = False,
    ) -> None:
        """Initialize total-return statistic algo."""
        super().__init__(factor_key="total_return", standardize=standardize)
        if not isinstance(lookback, pd.DateOffset):
            raise TypeError("TotalReturn `lookback` must be a pandas.DateOffset.")
        if not isinstance(lag, pd.DateOffset):
            raise TypeError("TotalReturn `lag` must be a pandas.DateOffset.")
        self.lookback = lookback
        self.lag = lag

    def calculate_factor(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
    ) -> pd.Series | None:
        """Compute total return for active names."""
        end = now - self.lag
        start = now - self.lookback
        try:
            prc = universe.loc[start:end, selected]
        except (TypeError, KeyError, ValueError):
            return None
        if prc.empty:
            return None

        prc = prc.ffill().bfill()
        first = prc.iloc[0].replace(0, pd.NA)
        last = prc.iloc[-1]
        return (last / first) - 1
