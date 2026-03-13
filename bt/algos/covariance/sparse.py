"""Sparse covariance estimators."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso

from bt.algos.covariance.core import Covariance
from utils.math_utils import validate_integer, validate_non_negative, validate_real


class GraphicalLassoCovariance(Covariance):
    """Estimate covariance with Graphical Lasso.

    Parameters
    ----------
    alpha : float, optional
        L1 regularization strength for inverse covariance estimation. Must be
        strictly positive.
    min_coverage : float, optional
        Minimum non-missing fraction required for an asset to be included in
        fitting. Must be in ``[0, 1]``.
    use_log_returns : bool, optional
        If ``True`` (default), transform returns via ``np.log1p`` before fit.
    max_iter : int, optional
        Maximum coordinate-descent iterations for solver convergence.
    tol : float, optional
        Solver tolerance. Must be strictly positive.
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        min_coverage: float = 0.8,
        use_log_returns: bool = True,
        max_iter: int = 200,
        tol: float = 1e-4,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        alpha_val = validate_real(alpha, "GraphicalLassoCovariance `alpha`")
        if alpha_val <= 0.0:
            raise ValueError("GraphicalLassoCovariance `alpha` must be > 0.")
        coverage = validate_non_negative(
            validate_real(min_coverage, "GraphicalLassoCovariance `min_coverage`"),
            "GraphicalLassoCovariance `min_coverage`",
        )
        if coverage > 1.0:
            raise ValueError("GraphicalLassoCovariance `min_coverage` must be <= 1.")
        if not isinstance(use_log_returns, bool):
            raise TypeError(
                "GraphicalLassoCovariance `use_log_returns` must be a bool."
            )
        iter_val = int(
            validate_integer(max_iter, "GraphicalLassoCovariance `max_iter`")
        )
        if (
            int(validate_non_negative(iter_val, "GraphicalLassoCovariance `max_iter`"))
            <= 0
        ):
            raise ValueError("GraphicalLassoCovariance `max_iter` must be > 0.")
        tol_val = validate_real(tol, "GraphicalLassoCovariance `tol`")
        if tol_val <= 0.0:
            raise ValueError("GraphicalLassoCovariance `tol` must be > 0.")

        self.alpha = alpha_val
        self.min_coverage = coverage
        self.use_log_returns = use_log_returns
        self.max_iter = iter_val
        self.tol = tol_val

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return sparse covariance matrix estimated by Graphical Lasso."""
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

        model_input = fit_data.astype(float)
        if self.use_log_returns:
            model_input = pd.DataFrame(
                np.log1p(model_input),
                index=model_input.index,
                columns=model_input.columns,
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
        if len(eligible_assets) == 1:
            variance = float(model_input.iloc[:, 0].var(ddof=1))
            return pd.DataFrame(
                [[variance]],
                index=eligible_assets,
                columns=eligible_assets,
                dtype=float,
            )

        try:
            estimator = GraphicalLasso(
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            estimator.fit(model_input.to_numpy(dtype=float))
        except Exception:
            return None

        return pd.DataFrame(
            estimator.covariance_,
            index=eligible_assets,
            columns=eligible_assets,
        )
