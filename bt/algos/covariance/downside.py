"""Downside-risk covariance estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bt.algos.covariance.core import Covariance
from utils.math_utils import validate_real


class SemiCovariance(Covariance):
    """Estimate downside semi-covariance around a threshold return.

    For each asset return series ``r``, the downside deviation is:
    ``min(r - threshold_return, 0)``.

    The estimator computes pairwise semi-covariance with pairwise-missing
    handling:
    - numerator: ``X.T @ X`` where missing values in ``X`` are treated as zero
    - denominator: pairwise overlap counts
    - output: ``numerator / denominator`` for valid overlaps

    Parameters
    ----------
    threshold_return : float, optional
        Per-period return threshold. Returns above this threshold contribute
        zero downside deviation.
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        threshold_return: float = 0.0,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        self.threshold_return = validate_real(
            threshold_return,
            "SemiCovariance `threshold_return`",
        )

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return downside semi-covariance matrix for selected assets.

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

        downside = (selected_returns - self.threshold_return).clip(upper=0.0)
        x = downside.fillna(0.0)
        overlap = downside.notna().astype(float)

        numerator = x.T @ x
        denominator = overlap.T @ overlap
        semi = numerator.divide(denominator.where(denominator > 0))
        semi = semi.reindex(index=selected, columns=selected)
        return semi.astype(float)
