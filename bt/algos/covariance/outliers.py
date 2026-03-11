"""Outlier-robust covariance estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

from bt.algos.covariance.core import Covariance
from utils.math_utils import (
    validate_integer,
    validate_non_negative,
    validate_real,
)


class MinCovDetCovariance(Covariance):
    """Estimate covariance with Minimum Covariance Determinant (MCD).

    Parameters
    ----------
    min_coverage : float, optional
        Minimum non-missing fraction required for an asset to be included.
        Assets below this threshold are excluded before model fitting.
    lookback : pandas.DateOffset, optional
        Historical window used to build returns.
    lag : pandas.DateOffset, optional
        Lag applied to the evaluation date.

    Notes
    -----
    This estimator is robust to outliers by fitting
    ``sklearn.covariance.MinCovDet`` on a subset of observations identified by
    the algorithm.
    """

    def __init__(
        self,
        min_coverage: float = 0.8,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        coverage = validate_non_negative(
            validate_real(min_coverage, "MinCovDetCovariance `min_coverage`"),
            "MinCovDetCovariance `min_coverage`",
        )
        if coverage > 1.0:
            raise ValueError("MinCovDetCovariance `min_coverage` must be <= 1.")
        self.min_coverage = coverage

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return MCD covariance matrix for selected assets.

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

        eligible_assets, fit_data = self._coverage_filtered_fit_data(
            selected_returns, self.min_coverage
        )
        if not eligible_assets:
            return pd.DataFrame()
        if fit_data.empty:
            return pd.DataFrame(
                index=eligible_assets, columns=eligible_assets, dtype=float
            )

        covariance = MinCovDet().fit(fit_data).covariance_
        return pd.DataFrame(covariance, index=eligible_assets, columns=eligible_assets)


class RobustHuberCovariance(Covariance):
    """Estimate covariance with iterative Huber downweighting.

    This estimator computes a robust location and covariance by iteratively
    downweighting observations with large standardized residuals.

    Parameters
    ----------
    c : float, optional
        Huber threshold controlling downweighting strength. Must be > 0.
    min_coverage : float, optional
        Minimum non-missing fraction required for an asset to be included in
        fitting. Must be in ``[0, 1]``.
    max_iter : int, optional
        Maximum number of robust-mean update iterations. Must be > 0.
    tol : float, optional
        Convergence tolerance for robust-mean updates. Must be > 0.
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        c: float = 1.345,
        min_coverage: float = 0.8,
        max_iter: int = 50,
        tol: float = 1e-6,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        c_val = validate_real(c, "RobustHuberCovariance `c`")
        if c_val <= 0.0:
            raise ValueError("RobustHuberCovariance `c` must be > 0.")
        coverage = validate_non_negative(
            validate_real(min_coverage, "RobustHuberCovariance `min_coverage`"),
            "RobustHuberCovariance `min_coverage`",
        )
        if coverage > 1.0:
            raise ValueError("RobustHuberCovariance `min_coverage` must be <= 1.")
        max_iter_val = int(
            validate_integer(max_iter, "RobustHuberCovariance `max_iter`")
        )
        if (
            int(validate_non_negative(max_iter_val, "RobustHuberCovariance `max_iter`"))
            <= 0
        ):
            raise ValueError("RobustHuberCovariance `max_iter` must be > 0.")
        tol_val = validate_real(tol, "RobustHuberCovariance `tol`")
        if tol_val <= 0.0:
            raise ValueError("RobustHuberCovariance `tol` must be > 0.")

        self.c = c_val
        self.min_coverage = coverage
        self.max_iter = max_iter_val
        self.tol = tol_val

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return Huber-robust covariance matrix for selected assets."""
        selected_returns = returns_history.reindex(columns=selected)
        if selected_returns.empty:
            return pd.DataFrame(index=selected, columns=selected, dtype=float)

        eligible_assets, fit_data = self._coverage_filtered_fit_data(
            selected_returns, self.min_coverage
        )
        if not eligible_assets:
            return pd.DataFrame()
        if fit_data.empty:
            return pd.DataFrame(
                index=eligible_assets, columns=eligible_assets, dtype=float
            )

        x = fit_data.to_numpy(dtype=float)
        n_obs, n_assets = x.shape
        if n_obs <= 1:
            return pd.DataFrame(
                index=eligible_assets, columns=eligible_assets, dtype=float
            )
        if n_assets == 1:
            variance = float(np.var(x[:, 0], ddof=1))
            return pd.DataFrame(
                [[variance]], index=eligible_assets, columns=eligible_assets
            )

        mu = np.median(x, axis=0)
        # Robust per-asset scale via MAD.
        mad = np.median(np.abs(x - mu), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-12)

        for _ in range(self.max_iter):
            z = (x - mu) / scale
            row_norm = np.sqrt(np.mean(z * z, axis=1))
            row_norm = np.maximum(row_norm, 1e-12)
            row_weight = np.minimum(1.0, self.c / row_norm)

            mu_new = np.average(x, axis=0, weights=row_weight)
            if np.linalg.norm(mu_new - mu) <= self.tol:
                mu = mu_new
                break
            mu = mu_new

        centered = x - mu
        weight_sum = float(np.sum(row_weight))
        denom = max(weight_sum - 1.0, 1.0)
        weighted = centered * row_weight[:, None]
        covariance = (weighted.T @ centered) / denom
        covariance = 0.5 * (covariance + covariance.T)
        return pd.DataFrame(covariance, index=eligible_assets, columns=eligible_assets)
