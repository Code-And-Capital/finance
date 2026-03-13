"""Shrinkage covariance estimators."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import OAS, ledoit_wolf

from bt.algos.covariance.core import Covariance
from utils.math_utils import validate_non_negative, validate_real


class OASCovariance(Covariance):
    """Estimate covariance with Oracle Approximating Shrinkage (OAS).

    This estimator first applies per-asset coverage filtering, then fits
    ``sklearn.covariance.OAS`` on complete-case rows for the remaining assets.

    Parameters
    ----------
    min_coverage : float, optional
        Minimum non-missing fraction required for an asset to be included in
        OAS fitting. Must be in ``[0, 1]``.
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        min_coverage: float = 0.8,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        coverage = validate_non_negative(
            validate_real(min_coverage, "OASCovariance `min_coverage`"),
            "OASCovariance `min_coverage`",
        )
        if coverage > 1.0:
            raise ValueError("OASCovariance `min_coverage` must be <= 1.")
        self.min_coverage = coverage

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return OAS covariance matrix for selected assets.

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

        covariance = OAS().fit(fit_data).covariance_
        return pd.DataFrame(covariance, index=eligible_assets, columns=eligible_assets)


class LedoitWolfCovariance(Covariance):
    """Estimate covariance with the Ledoit-Wolf shrinkage estimator.

    This estimator first applies per-asset coverage filtering, then fits the
    Ledoit-Wolf shrinkage estimator on complete-case rows for the remaining
    assets.

    Parameters
    ----------
    min_coverage : float, optional
        Minimum non-missing fraction required for an asset to be included in
        fitting. Must be in ``[0, 1]``.
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        min_coverage: float = 0.8,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        coverage = validate_non_negative(
            validate_real(min_coverage, "LedoitWolfCovariance `min_coverage`"),
            "LedoitWolfCovariance `min_coverage`",
        )
        if coverage > 1.0:
            raise ValueError("LedoitWolfCovariance `min_coverage` must be <= 1.")
        self.min_coverage = coverage

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return Ledoit-Wolf covariance matrix for selected assets.

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

        covariance = ledoit_wolf(fit_data)[0]
        return pd.DataFrame(covariance, index=eligible_assets, columns=eligible_assets)


class LedoitWolfNonLinearCovariance(Covariance):
    """Estimate covariance with nonlinear Ledoit-Wolf shrinkage (QIS style).

    This estimator applies a nonlinear shrinkage transformation to sample
    covariance eigenvalues (Quadratic-Inverse Shrinkage style), then
    reconstructs a covariance matrix in the original asset space.

    Parameters
    ----------
    min_coverage : float, optional
        Minimum non-missing fraction required for an asset to be included in
        fitting. Must be in ``[0, 1]``.
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        min_coverage: float = 0.8,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        coverage = validate_non_negative(
            validate_real(min_coverage, "LedoitWolfNonLinearCovariance `min_coverage`"),
            "LedoitWolfNonLinearCovariance `min_coverage`",
        )
        if coverage > 1.0:
            raise ValueError(
                "LedoitWolfNonLinearCovariance `min_coverage` must be <= 1."
            )
        self.min_coverage = coverage

    @staticmethod
    def _qis_covariance(fit_data: pd.DataFrame) -> pd.DataFrame:
        """Compute QIS-style nonlinear shrinkage covariance."""
        centered = fit_data.sub(fit_data.mean(axis=0), axis=1)
        t_obs, p = centered.shape
        n = t_obs - 1
        if n <= 0:
            return pd.DataFrame(
                index=fit_data.columns, columns=fit_data.columns, dtype=float
            )

        sample = centered.cov(ddof=1).to_numpy(dtype=float)
        eigvals, eigvecs = np.linalg.eigh(sample)
        eigvals = np.clip(eigvals, 0.0, None)

        c = p / n
        m = min(p, n)
        h = (min(c**2, 1.0 / (c**2)) ** 0.35) / (p**0.35)

        start_idx = max(1, p - n + 1) - 1
        eigvals_tail = np.maximum(eigvals[start_idx:], 1e-12)
        inv_lambda = 1.0 / eigvals_tail

        Lj = np.tile(inv_lambda.reshape(-1, 1), (1, m))
        Lj_i = Lj - Lj.T
        denominator = (Lj_i * Lj_i) + ((Lj * Lj) * (h**2))

        theta = np.mean((Lj * Lj_i) / denominator, axis=0)
        htheta = np.mean((Lj * (Lj * h)) / denominator, axis=0)
        atheta2 = theta**2 + htheta**2

        if p <= n:
            delta = 1.0 / (
                ((1.0 - c) ** 2) * inv_lambda
                + 2.0 * c * (1.0 - c) * inv_lambda * theta
                + (c**2) * inv_lambda * atheta2
            )
        else:
            delta0 = 1.0 / ((c - 1.0) * np.mean(inv_lambda))
            delta = np.concatenate(
                [
                    np.repeat(delta0, p - n),
                    1.0 / (inv_lambda * atheta2),
                ]
            )

        delta_sum = float(np.sum(delta))
        if delta_sum <= 0:
            return pd.DataFrame(
                index=fit_data.columns, columns=fit_data.columns, dtype=float
            )
        delta_qis = delta * (float(np.sum(eigvals)) / delta_sum)

        sigma_hat = eigvecs @ np.diag(delta_qis) @ eigvecs.T
        return pd.DataFrame(sigma_hat, index=fit_data.columns, columns=fit_data.columns)

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return nonlinear Ledoit-Wolf covariance matrix for selected assets."""
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

        fit_data = fit_data.astype(float)
        cov = self._qis_covariance(fit_data)
        return cov.reindex(index=eligible_assets, columns=eligible_assets)
