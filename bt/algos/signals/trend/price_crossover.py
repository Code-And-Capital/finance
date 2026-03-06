from typing import Any

import pandas as pd

from bt.algos.signals.core import Signal


class PriceCrossOverSignal(Signal):
    """Select names where price is above a reference series.

    Parameters
    ----------
    ma_name : str, optional
        Temp key containing reference levels as ``pandas.Series``.
    """

    def __init__(
        self,
        ma_name: str = "moving_average",
    ) -> None:
        """Initialize price-vs-reference trend signal."""
        super().__init__()
        if not isinstance(ma_name, str) or not ma_name:
            raise TypeError(
                "PriceCrossOverSignal `ma_name` must be a non-empty string."
            )
        self.ma_name = ma_name

    def _compute_signal(
        self,
        target: Any,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        candidate_pool: list[Any],
    ) -> pd.Series | None:
        """Return ``latest_prices > reference`` over the active asset set."""
        latest_prices = self._resolve_latest_prices(universe, now, candidate_pool)
        if latest_prices is None:
            return None
        if latest_prices.empty:
            return pd.Series(dtype=bool)

        ref = temp.get(self.ma_name)
        if not isinstance(ref, pd.Series):
            return None

        ref_on_assets = ref.reindex(latest_prices.index)
        return latest_prices > ref_on_assets


class DualMACrossoverSignal(Signal):
    """Select names where short moving average is above long moving average.

    Parameters
    ----------
    short_name : str, optional
        Temp key containing short-horizon moving-average ``Series``.
    long_name : str, optional
        Temp key containing long-horizon moving-average ``Series``.
    """

    def __init__(
        self,
        short_name: str = "ma_short",
        long_name: str = "ma_long",
    ) -> None:
        """Initialize dual moving-average crossover signal."""
        super().__init__()
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
        self,
        target: Any,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        candidate_pool: list[Any],
    ) -> pd.Series | None:
        """Return ``short_ma > long_ma`` over the active asset set."""
        short_ma = temp.get(self.short_name)
        long_ma = temp.get(self.long_name)
        if not isinstance(short_ma, pd.Series) or not isinstance(long_ma, pd.Series):
            return None

        short_on_assets = short_ma.reindex(candidate_pool)
        long_on_assets = long_ma.reindex(candidate_pool)
        return short_on_assets > long_on_assets
