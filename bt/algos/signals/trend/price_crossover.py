from typing import Any

import pandas as pd

from bt.algos.core import Algo
from utils.list_utils import keep_items_in_pool


class TrendSignalBase(Algo):
    """Base class for cross-sectional trend signals.

    This class handles common mechanics:
    - resolve current evaluation timestamp with optional lag
    - resolve candidate pool from ``temp["selected"]`` or full universe
    - fetch latest available price snapshot at/behind evaluation time
    - write selected tickers back to ``temp["selected"]`` as a list

    Subclasses implement ``_compute_signal`` and return a boolean ``Series``
    indexed by ticker.
    """

    def __init__(self, lag: pd.DateOffset = pd.DateOffset(days=0)) -> None:
        """Initialize the trend signal base."""
        super().__init__()
        if not isinstance(lag, pd.DateOffset):
            raise TypeError("TrendSignalBase `lag` must be a pandas.DateOffset.")
        self.lag = lag

    def __call__(self, target: Any) -> bool:
        """Compute and store selected tickers in ``target.temp['selected']``."""
        context = self._resolve_temp_universe_now(target)
        if context is None:
            return False
        temp, universe, now = context
        eval_now = now - self.lag

        if "selected" in temp:
            raw_selected = temp.get("selected")
            if not isinstance(raw_selected, list):
                return False
            candidate_pool = keep_items_in_pool(list(universe.columns), raw_selected)
        else:
            candidate_pool = list(universe.columns)

        if not candidate_pool:
            temp["selected"] = []
            return True

        try:
            price_history = universe.loc[:eval_now, candidate_pool]
        except (TypeError, KeyError, ValueError):
            return False
        if price_history.empty:
            return False

        latest_prices = price_history.iloc[-1].dropna()
        if latest_prices.empty:
            temp["selected"] = []
            return True

        signal = self._compute_signal(temp, latest_prices)
        if not isinstance(signal, pd.Series):
            return False

        # Normalize to boolean mask over latest-price names.
        signal = signal.reindex(latest_prices.index, fill_value=False)
        try:
            mask = signal.astype(bool)
        except (TypeError, ValueError):
            return False

        temp["selected"] = list(mask[mask].index)
        return True

    def _compute_signal(
        self, temp: dict[str, Any], latest_prices: pd.Series
    ) -> pd.Series | None:
        """Return boolean signal series indexed by asset names."""
        raise NotImplementedError("Subclasses must implement this method.")


class PriceCrossOverSignal(TrendSignalBase):
    """Select names where price is above a reference series.

    Parameters
    ----------
    ma_name : str, optional
        Temp key containing reference levels as ``pandas.Series``.
    lag : pandas.DateOffset, optional
        Time lag applied to ``target.now`` before evaluation.
    """

    def __init__(
        self,
        ma_name: str = "moving_average",
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        """Initialize price-vs-reference trend signal."""
        super().__init__(lag=lag)
        if not isinstance(ma_name, str) or not ma_name:
            raise TypeError(
                "PriceCrossOverSignal `ma_name` must be a non-empty string."
            )
        self.ma_name = ma_name

    def _compute_signal(
        self, temp: dict[str, Any], latest_prices: pd.Series
    ) -> pd.Series | None:
        """Return ``latest_prices > reference`` over the active asset set."""
        ref = temp.get(self.ma_name)
        if not isinstance(ref, pd.Series):
            return None

        ref_on_assets = ref.reindex(latest_prices.index)
        return latest_prices > ref_on_assets


class DualMACrossoverSignal(TrendSignalBase):
    """Select names where short moving average is above long moving average.

    Parameters
    ----------
    short_name : str, optional
        Temp key containing short-horizon moving-average ``Series``.
    long_name : str, optional
        Temp key containing long-horizon moving-average ``Series``.
    lag : pandas.DateOffset, optional
        Time lag applied to ``target.now`` before evaluation.
    """

    def __init__(
        self,
        short_name: str = "ma_short",
        long_name: str = "ma_long",
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        """Initialize dual moving-average crossover signal."""
        super().__init__(lag=lag)
        if not isinstance(short_name, str) or not short_name:
            raise TypeError(
                "DualMACrossoverSignal `short_name` must be a non-empty string."
            )
        if not isinstance(long_name, str) or not long_name:
            raise TypeError(
                "DualMACrossoverSignal `long_name` must be a non-empty string."
            )
        self.short_name = short_name
        self.long_name = long_name

    def _compute_signal(
        self, temp: dict[str, Any], latest_prices: pd.Series
    ) -> pd.Series | None:
        """Return ``short_ma > long_ma`` over the active asset set."""
        short_ma = temp.get(self.short_name)
        long_ma = temp.get(self.long_name)
        if not isinstance(short_ma, pd.Series) or not isinstance(long_ma, pd.Series):
            return None

        short_on_assets = short_ma.reindex(latest_prices.index)
        long_on_assets = long_ma.reindex(latest_prices.index)
        return short_on_assets > long_on_assets
