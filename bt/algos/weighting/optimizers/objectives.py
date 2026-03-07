"""Objective builders for optimization problems."""

from __future__ import annotations

import numpy as np
import cvxpy as cvx


def negative_sharpe_ratio(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    rf: float = 0.0,
) -> float:
    """Return negative Sharpe ratio used in mean-variance optimization."""
    port_mean = float(np.dot(expected_returns, weights))
    port_var = float(np.dot(weights, covariance.dot(weights)))
    if port_var <= 0.0 or not np.isfinite(port_var):
        return float("inf")
    return -(port_mean - rf) / np.sqrt(port_var)


def mean_variance_utility_objective(
    weights: cvx.Variable,
    expected_returns: cvx.Parameter,
    covariance,
    risk_averse_lambda: float,
    maximize_builder,
):
    """Build CVXPY maximize objective for mean-variance utility."""
    return_term = weights.T @ expected_returns
    risk_term = cvx.quad_form(weights, covariance)
    return maximize_builder(return_term - risk_averse_lambda * risk_term)


def risk_parity_objective(
    weights: cvx.Variable,
    covariance,
    risk_budgets: np.ndarray,
    minimize_builder,
):
    """Build CVXPY minimize objective for risk-parity optimization."""
    return minimize_builder(
        0.5 * cvx.quad_form(weights, covariance) - risk_budgets @ cvx.log(weights)
    )
