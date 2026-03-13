"""Variable and parameter builders for optimizers."""

from collections.abc import Callable
from typing import Any

import cvxpy as cvx
import pandas as pd


def build_covariance_matrix(covariance: pd.DataFrame):
    """Build PSD-safe covariance matrix expression for optimization."""
    return cvx.atoms.affine.wraps.psd_wrap(covariance.to_numpy(dtype=float))


def build_expected_returns_parameter(
    returns: pd.Series,
    parameter_builder: Callable[..., Any],
):
    """Build and set expected-returns CVXPY parameter."""
    parameter = parameter_builder(shape=len(returns))
    parameter.value = (
        pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    return parameter


def build_weight_variable(
    asset_count: int,
    variable_builder: Callable[..., Any],
):
    """Build optimization weight variable."""
    return variable_builder(shape=asset_count)
