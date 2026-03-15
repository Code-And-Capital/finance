"""Economic data query helpers for revision-aware FRED time series."""

from typing import Optional, Sequence

import pandas as pd

from handyman.base import DateLike, read_table_by_filters
from utils.dataframe_utils import ensure_datetime_column


def get_economic_metadata(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
) -> pd.DataFrame:
    """Return FRED metadata rows with optional ticker/date filters.

    Parameters
    ----------
    tickers
        Optional FRED ticker filter value(s).
    start_date
        Optional inclusive lower bound applied to ``OBSERVATION_START``.
    end_date
        Optional inclusive upper bound applied to ``OBSERVATION_START``.
    configs_path
        Optional path to the configuration file with Azure credentials.

    Returns
    -------
    pandas.DataFrame
        Metadata rows with ``OBSERVATION_START`` coerced to pandas datetime
        when present.
    """
    df = read_table_by_filters(
        table_name="fred_series",
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        configs_path=configs_path,
        date_column="OBSERVATION_START",
    )
    if "OBSERVATION_START" in df.columns:
        df = ensure_datetime_column(df, "OBSERVATION_START")
    return df


def get_economic_data(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
) -> pd.DataFrame:
    """Return revision-aware FRED rows with optional ticker/date filters.

    Parameters
    ----------
    tickers
        Optional FRED ticker filter value(s).
    start_date
        Optional inclusive lower bound applied to ``OBSERVATION_DATE``.
    end_date
        Optional inclusive upper bound applied to ``OBSERVATION_DATE``.
    configs_path
        Optional path to the configuration file with Azure credentials.

    Returns
    -------
    pandas.DataFrame
        Economic data rows with ``OBSERVATION_DATE``, ``REALTIME_START``, and
        ``REALTIME_END`` coerced to pandas datetime when present.
    """
    df = read_table_by_filters(
        table_name="fred_economic_data",
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        configs_path=configs_path,
        date_column="OBSERVATION_DATE",
    )
    df = ensure_datetime_column(df, "OBSERVATION_DATE")
    for column in ["REALTIME_START", "REALTIME_END"]:
        if column in df.columns:
            df = ensure_datetime_column(df, column)
    return df
