"""Likelihood-based covariance estimators."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance as SklearnEmpiricalCovariance

from bt.algos.covariance.core import Covariance
from utils.math_utils import validate_non_negative, validate_real


class EmpiricalCovariance(Covariance):
    """Estimate covariance with sklearn's empirical MLE estimator.

    This estimator applies an optional ``log1p`` transform to returns and then
    fits ``sklearn.covariance.EmpiricalCovariance``.

    Parameters
    ----------
    min_coverage : float, optional
        Minimum non-missing fraction required for an asset to be included in
        model fitting. Must be in ``[0, 1]``.
    use_log_returns : bool, optional
        If ``True`` (default), transform returns with ``np.log1p`` before fit.
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        min_coverage: float = 0.8,
        use_log_returns: bool = True,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        coverage = validate_non_negative(
            validate_real(min_coverage, "EmpiricalCovariance `min_coverage`"),
            "EmpiricalCovariance `min_coverage`",
        )
        if coverage > 1.0:
            raise ValueError("EmpiricalCovariance `min_coverage` must be <= 1.")
        if not isinstance(use_log_returns, bool):
            raise TypeError("EmpiricalCovariance `use_log_returns` must be a bool.")

        self.min_coverage = coverage
        self.use_log_returns = use_log_returns

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return empirical covariance matrix for selected assets."""
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

        if self.use_log_returns:
            fit_data = pd.DataFrame(
                np.log1p(fit_data),
                index=fit_data.index,
                columns=fit_data.columns,
            )

        covariance = SklearnEmpiricalCovariance().fit(fit_data).covariance_
        return pd.DataFrame(covariance, index=eligible_assets, columns=eligible_assets)
