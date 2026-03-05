from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bt.algos.factors.core.factor import Factor
from utils.math_utils import (
    validate_integer,
    validate_non_negative,
    validate_real,
)


class MovingAverage(Factor):
    """Base class for cross-sectional moving-average estimators.

    Parameters
    ----------
    lag : pandas.DateOffset, optional
        Time lag applied to ``target.now`` before evaluating the indicator.
    factor_key : str, optional
        Key used to store the computed cross-section in ``target.temp``.
    """

    def __init__(
        self,
        lag: pd.DateOffset = pd.DateOffset(days=0),
        factor_key: str = "moving_average",
    ) -> None:
        """Initialize the moving-average base class."""
        if not isinstance(lag, pd.DateOffset):
            raise TypeError("MovingAverage `lag` must be a pandas.DateOffset.")
        if not isinstance(factor_key, str) or not factor_key:
            raise TypeError("MovingAverage `factor_key` must be a non-empty string.")
        super().__init__(factor_key=factor_key)
        self.lag = lag

    def calculate_factor(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
    ) -> pd.Series | None:
        """Compute moving-average levels for active names."""
        end = now - self.lag

        try:
            hist = universe.loc[:end, selected]
        except (TypeError, KeyError, ValueError):
            return None
        if hist.empty:
            return None

        ma = self._compute_average(hist, now)
        if not isinstance(ma, pd.Series):
            return None

        return ma.reindex(selected)

    def _compute_average(self, hist: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
        """Return moving-average levels for the given history window."""
        raise NotImplementedError("Subclasses must implement `_compute_average`.")


class SimpleMovingAverage(MovingAverage):
    """Compute simple moving averages over a lookback window.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Historical window length ending at ``end``.
    measure : {"mean", "median"}, optional
        Aggregation function over the lookback window.
    lag : pandas.DateOffset, optional
        Lag applied to ``target.now`` before evaluation.
    factor_key : str, optional
        Output key in ``target.temp``.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        measure: str = "mean",
        lag: pd.DateOffset = pd.DateOffset(days=0),
        factor_key: str = "moving_average",
    ) -> None:
        """Initialize simple moving-average estimator."""
        super().__init__(lag=lag, factor_key=factor_key)
        if not isinstance(lookback, pd.DateOffset):
            raise TypeError(
                "SimpleMovingAverage `lookback` must be a pandas.DateOffset."
            )
        if measure not in {"mean", "median"}:
            raise ValueError(
                "SimpleMovingAverage `measure` must be 'mean' or 'median'."
            )
        self.lookback = lookback
        self.measure = measure

    def _compute_average(self, hist: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
        """Compute simple moving-average cross-section from ``now - lookback`` onward."""
        start = now - self.lookback
        window = hist.loc[start:]
        if window.empty:
            return pd.Series(dtype=float)
        if self.measure == "mean":
            return window.mean()
        return window.median()


class ExponentialWeightedMovingAverage(MovingAverage):
    """Compute exponentially weighted moving averages.

    Parameters
    ----------
    half_life : float
        Positive half-life parameter expressed in periods.
    lag : pandas.DateOffset, optional
        Lag applied to ``target.now`` before evaluation.
    factor_key : str, optional
        Output key in ``target.temp``.
    """

    def __init__(
        self,
        half_life: float,
        lag: pd.DateOffset = pd.DateOffset(days=0),
        factor_key: str = "ewma",
    ) -> None:
        """Initialize EWMA estimator."""
        super().__init__(lag=lag, factor_key=factor_key)
        half_life_val = validate_non_negative(
            validate_real(half_life, "ExponentialWeightedMovingAverage `half_life`"),
            "ExponentialWeightedMovingAverage `half_life`",
        )
        if half_life_val == 0:
            raise ValueError(
                "ExponentialWeightedMovingAverage `half_life` must be > 0."
            )
        self.half_life = half_life_val
        self.alpha = 1.0 - 2.0 ** (-1.0 / self.half_life)

    def _compute_average(self, hist: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
        """Compute EWMA cross-section from available history."""
        ewma = hist.ewm(alpha=self.alpha, adjust=False, min_periods=1).mean()
        return ewma.iloc[-1]


class KernelMovingAverage(MovingAverage):
    """Compute rolling kernel-weighted moving averages.

    The kernel is polynomial in recency:
    ``w_t ∝ t**kernel_factor`` for ``t = 1..lookback_periods`` where larger
    ``t`` is more recent data.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Historical window length ending at ``now - lag``.
    kernel_factor : int, optional
        Non-negative kernel exponent:
        - 0: uniform kernel
        - 1: linear kernel
        - >1: increasingly recency-biased
    lag : pandas.DateOffset, optional
        Lag applied to ``target.now`` before evaluation.
    factor_key : str, optional
        Output key in ``target.temp``.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        kernel_factor: int = 1,
        lag: pd.DateOffset = pd.DateOffset(days=0),
        factor_key: str = "kernel_moving_average",
    ) -> None:
        """Initialize kernel moving-average estimator."""
        super().__init__(lag=lag, factor_key=factor_key)
        if not isinstance(lookback, pd.DateOffset):
            raise TypeError(
                "KernelMovingAverage `lookback` must be a pandas.DateOffset."
            )
        kernel_val = validate_integer(
            kernel_factor, "KernelMovingAverage `kernel_factor`"
        )
        kernel_val = int(
            validate_non_negative(kernel_val, "KernelMovingAverage `kernel_factor`")
        )
        self.lookback = lookback
        self.kernel_factor = kernel_val

    def _compute_average(self, hist: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
        """Compute kernel-weighted moving-average cross-section."""
        start = now - self.lookback
        window = hist.loc[start:]
        if window.empty:
            return pd.Series(dtype=float)

        n_obs = len(window.index)
        recency = np.arange(1, n_obs + 1, dtype=float)
        kernel = recency**self.kernel_factor
        kernel = kernel / kernel.sum()

        def _weighted_col(col: pd.Series) -> float:
            values = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            if not finite.any():
                return np.nan
            w = kernel[finite]
            return float(np.average(values[finite], weights=w / w.sum()))

        return window.apply(_weighted_col, axis=0)
