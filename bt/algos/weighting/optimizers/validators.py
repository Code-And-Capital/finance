"""Validation helpers for optimization inputs."""

from __future__ import annotations

import pandas as pd


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
