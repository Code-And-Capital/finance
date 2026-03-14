import pandas as pd
import numpy as np
from utils.date_utils import coerce_timestamp


def convert_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt numeric conversion for each DataFrame column."""
    out = df.copy()
    for column in out.columns:
        try:
            out[column] = pd.to_numeric(out[column])
        except (ValueError, TypeError):
            pass
    return out


def df_to_dict(df: pd.DataFrame, key_col, value_col) -> dict:
    """Convert two columns of a DataFrame into a dictionary."""
    keys = df[key_col]
    values = df[value_col]
    return {(None if pd.isna(key) else key): value for key, value in zip(keys, values)}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase and underscore-normalize DataFrame columns."""
    out = df.copy()
    out.columns = [str(column).upper().replace(" ", "_") for column in out.columns]
    return out


def coerce_datetime_series(
    series: pd.Series,
    *,
    errors: str = "raise",
) -> pd.Series:
    """Coerce a Series to timezone-naive datetimes without UTC conversion."""

    def _convert(value):
        if pd.isna(value):
            return pd.NaT
        try:
            parsed = pd.Timestamp(value)
        except Exception:
            if errors == "raise":
                raise
            return pd.NaT
        if parsed.tzinfo is not None:
            parsed = parsed.tz_localize(None)
        return parsed

    converted = pd.Series((_convert(value) for value in series), index=series.index)
    return pd.to_datetime(converted, errors="coerce")


def coerce_numeric_series(
    series: pd.Series,
    label: str,
    value_name: str,
) -> pd.Series:
    """Return a finite float Series or raise a descriptive validation error."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(
            f"{label} `{value_name}` contains missing or non-numeric values."
        )
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} `{value_name}` must contain only finite values.")
    return pd.Series(values, index=series.index, dtype=float)


def coerce_numeric_frame(
    frame: pd.DataFrame,
    label: str,
    value_name: str,
) -> pd.DataFrame:
    """Return a finite float DataFrame or raise a descriptive validation error."""
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(
            f"{label} `{value_name}` contains missing or non-numeric values."
        )
    values = numeric.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} `{value_name}` must contain only finite values.")
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def ensure_datetime_column(df: pd.DataFrame, column: str = "DATE") -> pd.DataFrame:
    """Validate and coerce a column to pandas datetime."""
    if column not in df.columns:
        raise ValueError(f"Expected column '{column}' in SQL results")

    out = df.copy()
    out[column] = coerce_datetime_series(out[column], errors="raise")
    return out


def one_column_frame_to_series(frame: pd.DataFrame) -> pd.Series:
    """Return the single column in a DataFrame as a Series.

    Parameters
    ----------
    frame : pandas.DataFrame
        Input DataFrame that must have exactly one column.

    Returns
    -------
    pandas.Series
        The sole column as a Series.

    Raises
    ------
    ValueError
        If ``frame`` does not have exactly one column.
    """
    if frame.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one column.")
    return frame.iloc[:, 0]


def normalize_date_series(series: pd.Series, label: str = "date") -> pd.Series:
    """Coerce all values in a Series to validated pandas timestamps.

    Parameters
    ----------
    series : pandas.Series
        Input Series containing date-like values.
    label : str, optional
        Field label used in timestamp coercion error messages.

    Returns
    -------
    pandas.Series
        Series with the same index and ``pandas.Timestamp`` values.
    """
    return pd.Series(
        [coerce_timestamp(value, label) for value in series.values],
        index=series.index,
    )
