"""Validation helpers for optimization inputs."""

import pandas as pd
from utils.math_utils import validate_non_negative, validate_real


def validate_series(value, label: str, value_name: str) -> None:
    """Validate that ``value`` is a pandas Series."""
    if not isinstance(value, pd.Series):
        raise TypeError(f"{label} `{value_name}` must be a Series.")


def validate_bounds(
    bounds, label: str, value_name: str = "bounds"
) -> tuple[float, float]:
    """Validate and normalize a 2-tuple of non-negative lower/upper bounds."""
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise TypeError(f"{label} `{value_name}` must be a 2-tuple.")
    lower = validate_non_negative(
        validate_real(bounds[0], f"{value_name}[0]"),
        f"{value_name}[0]",
    )
    upper = validate_non_negative(
        validate_real(bounds[1], f"{value_name}[1]"),
        f"{value_name}[1]",
    )
    if lower > upper:
        raise ValueError(f"{label} `{value_name}` requires lower <= upper.")
    return (lower, upper)


def validate_square_covariance_matrix(covariance: pd.DataFrame, label: str) -> None:
    """Validate covariance matrix shape and axis alignment.

    Parameters
    ----------
    covariance
        Candidate covariance matrix to validate.
    label
        Name used in exception messages (usually optimizer class name).
    """
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError(f"{label} `covariance` must be a DataFrame.")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{label} covariance must be square.")
    if not covariance.index.equals(covariance.columns):
        raise ValueError(f"{label} covariance index/columns must match.")


def resolve_selected_covariance(
    covariance,
    selected: list[str],
) -> pd.DataFrame:
    """Return covariance subset for selected names.

    Returns
    -------
    pandas.DataFrame
        Selected covariance subset, or an empty DataFrame when subsetting fails.
    """
    try:
        return covariance.loc[selected, selected]
    except (KeyError, TypeError, ValueError):
        return pd.DataFrame()
