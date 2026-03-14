from typing import Any

import pandas as pd

from bt.algos.factors import ExponentialWeightedMovingAverage, SimpleMovingAverage
from bt.algos.signals.core import Signal


def _build_moving_average_algo(
    ma_type: str,
    *,
    factor_key: str | None = None,
    lag: pd.DateOffset = pd.DateOffset(days=0),
    lookback: pd.DateOffset = pd.DateOffset(months=3),
    measure: str = "mean",
    half_life: float | None = None,
) -> SimpleMovingAverage | ExponentialWeightedMovingAverage:
    """Build a supported moving-average factor from a string config."""
    if not isinstance(ma_type, str) or not ma_type:
        raise TypeError("Moving-average `ma_type` must be a non-empty string.")
    if ma_type == "simple":
        return SimpleMovingAverage(
            lookback=lookback,
            measure=measure,
            lag=lag,
            factor_key=factor_key or "moving_average",
        )
    if ma_type == "exponential":
        if half_life is None:
            raise TypeError(
                "Exponential moving average requires a non-null `half_life`."
            )
        return ExponentialWeightedMovingAverage(
            half_life=half_life,
            lag=lag,
            factor_key=factor_key or "ewma",
        )
    raise ValueError("Moving-average `ma_type` must be 'simple' or 'exponential'.")


class PriceCrossOverSignal(Signal):
    """Select names where price is above a reference series.

    Parameters
    ----------
    ma_type : {"simple", "exponential"}
        Type of moving average to compute internally.
    lookback : pandas.DateOffset, optional
        Lookback window for the simple moving average.
    measure : {"mean", "median"}, optional
        Aggregation used by the simple moving average.
    lag : pandas.DateOffset, optional
        Lag applied to the internal moving-average factor.
    half_life : float | None, optional
        Half-life used by the exponential moving average.
    """

    def __init__(
        self,
        ma_type: str,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        measure: str = "mean",
        lag: pd.DateOffset = pd.DateOffset(days=0),
        half_life: float | None = None,
    ) -> None:
        """Initialize price-vs-reference trend signal."""
        super().__init__()
        self.ma_algo = _build_moving_average_algo(
            ma_type,
            factor_key=None,
            lag=lag,
            lookback=lookback,
            measure=measure,
            half_life=half_life,
        )
        self.ma_name = self.ma_algo.factor_key
        self._register_factor_stats(self.ma_name, self.ma_algo.stats)

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

        temp["selected"] = list(candidate_pool)
        if not self.ma_algo(target):
            return None

        ref = temp.get(self.ma_name)
        if not isinstance(ref, pd.Series):
            return None

        ref_on_assets = ref.reindex(latest_prices.index)
        return latest_prices > ref_on_assets


class DualMACrossoverSignal(Signal):
    """Select names where short moving average is above long moving average.

    Parameters
    ----------
    short_ma_type : {"simple", "exponential"}
        Type used for the short-horizon moving average.
    long_ma_type : {"simple", "exponential"}
        Type used for the long-horizon moving average.
    short_lookback : pandas.DateOffset, optional
        Lookback window for a simple short moving average.
    long_lookback : pandas.DateOffset, optional
        Lookback window for a simple long moving average.
    short_measure : {"mean", "median"}, optional
        Aggregation used by a simple short moving average.
    long_measure : {"mean", "median"}, optional
        Aggregation used by a simple long moving average.
    short_lag : pandas.DateOffset, optional
        Lag applied to the short moving-average factor.
    long_lag : pandas.DateOffset, optional
        Lag applied to the long moving-average factor.
    short_half_life : float | None, optional
        Half-life used by an exponential short moving average.
    long_half_life : float | None, optional
        Half-life used by an exponential long moving average.
    """

    def __init__(
        self,
        short_ma_type: str,
        long_ma_type: str,
        short_lookback: pd.DateOffset = pd.DateOffset(months=1),
        long_lookback: pd.DateOffset = pd.DateOffset(months=3),
        short_measure: str = "mean",
        long_measure: str = "mean",
        short_lag: pd.DateOffset = pd.DateOffset(days=0),
        long_lag: pd.DateOffset = pd.DateOffset(days=0),
        short_half_life: float | None = None,
        long_half_life: float | None = None,
    ) -> None:
        """Initialize dual moving-average crossover signal."""
        super().__init__()
        self.short_ma_algo = _build_moving_average_algo(
            short_ma_type,
            factor_key="short",
            lag=short_lag,
            lookback=short_lookback,
            measure=short_measure,
            half_life=short_half_life,
        )
        self.long_ma_algo = _build_moving_average_algo(
            long_ma_type,
            factor_key="long",
            lag=long_lag,
            lookback=long_lookback,
            measure=long_measure,
            half_life=long_half_life,
        )
        self.short_name = self.short_ma_algo.factor_key
        self.long_name = self.long_ma_algo.factor_key
        self._register_factor_stats(self.short_name, self.short_ma_algo.stats)
        self._register_factor_stats(self.long_name, self.long_ma_algo.stats)

    def _compute_signal(
        self,
        target: Any,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        candidate_pool: list[Any],
    ) -> pd.Series | None:
        """Return ``short_ma > long_ma`` over the active asset set."""
        temp["selected"] = list(candidate_pool)
        if not self.short_ma_algo(target):
            return None
        temp["selected"] = list(candidate_pool)
        if not self.long_ma_algo(target):
            return None

        short_ma = temp.get(self.short_name)
        long_ma = temp.get(self.long_name)
        if not isinstance(short_ma, pd.Series) or not isinstance(long_ma, pd.Series):
            return None

        short_on_assets = short_ma.reindex(candidate_pool)
        long_on_assets = long_ma.reindex(candidate_pool)
        return short_on_assets > long_on_assets
