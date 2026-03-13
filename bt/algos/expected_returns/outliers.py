"""Outlier-robust expected-return estimators."""

from typing import Any

import pandas as pd

from utils.math_utils import validate_non_negative, validate_real

from .core import ExpectedReturns


def _validate_tail_fraction(value: float, label: str) -> float:
    """Validate symmetric tail fraction used for trimming/winsorization."""
    frac = validate_non_negative(validate_real(value, label), label)
    if frac >= 0.5:
        raise ValueError(f"{label} must be < 0.5.")
    return frac


class TrimmedMeanReturn(ExpectedReturns):
    """Estimate expected returns using quantile-trimmed mean returns.

    For each asset, returns below the ``trim_fraction`` quantile and above
    ``1 - trim_fraction`` quantile are removed before averaging.

    Parameters
    ----------
    trim_fraction : float, optional
        Symmetric tail fraction removed from each side. Must be in ``[0, 0.5)``.
    lookback : pandas.DateOffset, optional
        Historical lookback window used to build returns history.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        trim_fraction: float = 0.1,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        self.trim_fraction = _validate_tail_fraction(
            trim_fraction, "TrimmedMeanReturn `trim_fraction`"
        )

    def calculate_expected_returns(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.Series | None:
        """Return trimmed-mean return per selected asset."""
        out: dict[Any, float] = {}
        frame = returns_history.reindex(columns=selected)
        for asset in selected:
            series = pd.to_numeric(frame[asset], errors="coerce").dropna()
            if series.empty:
                out[asset] = float("nan")
                continue
            lower = float(series.quantile(self.trim_fraction))
            upper = float(series.quantile(1.0 - self.trim_fraction))
            trimmed = series[(series >= lower) & (series <= upper)]
            out[asset] = float(trimmed.mean()) if not trimmed.empty else float("nan")
        return pd.Series(out, index=selected, dtype=float)


class WinsorizedMeanReturn(ExpectedReturns):
    """Estimate expected returns using quantile-winsorized mean returns.

    For each asset, returns are clipped to the interval
    ``[q(trim_fraction), q(1-trim_fraction)]`` before averaging.

    Parameters
    ----------
    trim_fraction : float, optional
        Symmetric tail fraction for clipping. Must be in ``[0, 0.5)``.
    lookback : pandas.DateOffset, optional
        Historical lookback window used to build returns history.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        trim_fraction: float = 0.1,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        self.trim_fraction = _validate_tail_fraction(
            trim_fraction, "WinsorizedMeanReturn `trim_fraction`"
        )

    def calculate_expected_returns(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.Series | None:
        """Return winsorized-mean return per selected asset."""
        out: dict[Any, float] = {}
        frame = returns_history.reindex(columns=selected)
        for asset in selected:
            series = pd.to_numeric(frame[asset], errors="coerce").dropna()
            if series.empty:
                out[asset] = float("nan")
                continue
            lower = float(series.quantile(self.trim_fraction))
            upper = float(series.quantile(1.0 - self.trim_fraction))
            wins = series.clip(lower=lower, upper=upper)
            out[asset] = float(wins.mean())
        return pd.Series(out, index=selected, dtype=float)
