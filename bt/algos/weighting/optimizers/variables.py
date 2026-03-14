"""Variable and parameter builders for optimizers."""

from collections.abc import Callable
from typing import Any

import cvxpy as cvx
import pandas as pd


def build_covariance_matrix(covariance: pd.DataFrame):
    """Build PSD-safe covariance matrix expression for optimization."""
    return cvx.atoms.affine.wraps.psd_wrap(covariance.to_numpy(dtype=float))


def build_series_parameter(
    values: pd.Series,
    parameter_builder: Callable[..., Any],
):
    """Build and set a numeric CVXPY parameter from a Series."""
    parameter = parameter_builder(shape=len(values))
    parameter.value = (
        pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    return parameter


def build_matrix_parameter(
    values: pd.DataFrame,
    parameter_builder: Callable[..., Any],
):
    """Build and set a numeric CVXPY parameter from a DataFrame."""
    parameter = parameter_builder(shape=values.shape)
    parameter.value = values.to_numpy(dtype=float, copy=True)
    return parameter


def build_weight_variable(
    asset_count: int,
    variable_builder: Callable[..., Any],
):
    """Build optimization weight variable."""
    return variable_builder(shape=asset_count)
