"""GARCH-based covariance estimators."""

from __future__ import annotations

from typing import Any

import mgarch
import numpy as np
import pandas as pd

from bt.algos.covariance.core import Covariance
from utils.math_utils import validate_integer, validate_non_negative, validate_real


class GARCHCovariance(Covariance):
    """Estimate covariance using a multivariate GARCH model.

    The estimator fits ``mgarch`` on (optionally transformed) return history
    and forecasts the covariance matrix ``forecast_period`` steps ahead.

    Parameters
    ----------
    distribution : str, optional
        Distribution passed to ``mgarch.mgarch``. Supported values are
        ``"norm"`` and ``"t"``.
    forecast_period : int, optional
        Forecast horizon passed to ``model.predict``. Must be strictly positive.
    min_coverage : float, optional
        Minimum non-missing fraction required for an asset to be included in
        model fitting. Must be in ``[0, 1]``.
    use_log_returns : bool, optional
        If ``True`` (default), transform returns via ``np.log1p`` before fit.
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.

    Notes
    -----
    If model fit or prediction fails, this estimator returns ``None`` and the
    outer covariance call returns ``False``.
    """

    def __init__(
        self,
        distribution: str = "norm",
        forecast_period: int = 21,
        min_coverage: float = 0.8,
        use_log_returns: bool = True,
        lookback: pd.DateOffset = pd.DateOffset(years=2),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        if distribution not in {"norm", "t"}:
            raise ValueError("GARCHCovariance `distribution` must be 'norm' or 't'.")
        horizon = int(
            validate_integer(forecast_period, "GARCHCovariance `forecast_period`")
        )
        if (
            int(validate_non_negative(horizon, "GARCHCovariance `forecast_period`"))
            <= 0
        ):
            raise ValueError("GARCHCovariance `forecast_period` must be > 0.")
        coverage = validate_non_negative(
            validate_real(min_coverage, "GARCHCovariance `min_coverage`"),
            "GARCHCovariance `min_coverage`",
        )
        if coverage > 1.0:
            raise ValueError("GARCHCovariance `min_coverage` must be <= 1.")
        if not isinstance(use_log_returns, bool):
            raise TypeError("GARCHCovariance `use_log_returns` must be a bool.")

        self.distribution = distribution
        self.forecast_period = horizon
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
        """Return GARCH-forecast covariance matrix for selected assets."""
        selected_returns = returns_history.reindex(columns=selected)
        if selected_returns.empty:
            return pd.DataFrame(index=selected, columns=selected, dtype=float)

        eligible_assets, fit_data = self._coverage_filtered_fit_data(
            selected_returns,
            self.min_coverage,
        )
        if not eligible_assets:
            return pd.DataFrame()
        if fit_data.empty:
            return pd.DataFrame(
                index=eligible_assets, columns=eligible_assets, dtype=float
            )

        model_input = fit_data
        if self.use_log_returns:
            model_input = pd.DataFrame(
                np.log1p(fit_data),
                index=fit_data.index,
                columns=fit_data.columns,
            )
            model_input = model_input.replace([np.inf, -np.inf], np.nan).dropna(
                how="any"
            )
            if model_input.empty:
                return pd.DataFrame(
                    index=eligible_assets,
                    columns=eligible_assets,
                    dtype=float,
                )

        try:
            model = mgarch.mgarch(dist=self.distribution)
            model.fit(model_input.to_numpy(dtype=float))
            forecast = model.predict(self.forecast_period)
        except Exception:
            return None

        cov_matrix = forecast.get("cov") if isinstance(forecast, dict) else None
        if cov_matrix is None:
            return None
        cov_array = np.asarray(cov_matrix, dtype=float)
        n_assets = len(eligible_assets)
        if cov_array.shape != (n_assets, n_assets):
            return None
        return pd.DataFrame(cov_array, index=eligible_assets, columns=eligible_assets)
