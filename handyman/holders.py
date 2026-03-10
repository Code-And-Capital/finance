from __future__ import annotations

"""Holder query helpers for institutional and major holder datasets."""

from typing import Optional, Sequence

import pandas as pd

from handyman.base import (
    DateLike,
    build_security_filter_sql,
    read_table_by_filters,
    run_sql_template,
)
from sql.script_factory import default_sql_client
from utils.dataframe_utils import ensure_datetime_column


def get_institutional_holders(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return institutional holder rows with optional ticker/date filters.

    Parameters
    ----------
    tickers
        Optional ticker filter.
    figis
        Optional FIGI filter value(s). Cannot be combined with ``tickers``.
    start_date
        Optional inclusive lower-bound filter for ``DATE``.
    configs_path
        Optional path to the configuration file with Azure credentials.
    get_latest
        If True, return only rows from the latest snapshot date per ticker and
        ignore ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Institutional holder rows with date columns coerced to datetime.
    """
    if not get_latest:
        df = read_table_by_filters(
            table_name="institutional_holders",
            tickers=tickers,
            figis=figis,
            start_date=start_date,
            configs_path=configs_path,
        )
    else:
        security_filter = build_security_filter_sql(
            sql_client=default_sql_client,
            tickers=tickers,
            figis=figis,
        )
        df = run_sql_template(
            sql_file="institutional_holders_latest.txt",
            filters={"security_filter": security_filter, "date_filter": ""},
            configs_path=configs_path,
        )
    for column in ["DATE", "DATE_REPORTED"]:
        if column in df.columns:
            df = ensure_datetime_column(df, column)
    return df


def get_major_holders(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return major holder rows with optional ticker/date filters.

    Parameters
    ----------
    tickers
        Optional ticker filter.
    figis
        Optional FIGI filter value(s). Cannot be combined with ``tickers``.
    start_date
        Optional inclusive lower-bound filter for ``DATE``.
    configs_path
        Optional path to the configuration file with Azure credentials.
    get_latest
        If True, return only rows from the latest snapshot date per ticker and
        ignore ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Major holder rows with ``DATE`` coerced to datetime.
    """
    if not get_latest:
        df = read_table_by_filters(
            table_name="major_holders",
            tickers=tickers,
            figis=figis,
            start_date=start_date,
            configs_path=configs_path,
        )
    else:
        security_filter = build_security_filter_sql(
            sql_client=default_sql_client,
            tickers=tickers,
            figis=figis,
        )
        df = run_sql_template(
            sql_file="major_holders_latest.txt",
            filters={"security_filter": security_filter, "date_filter": ""},
            configs_path=configs_path,
        )
    if "DATE" in df.columns:
        df = ensure_datetime_column(df, "DATE")
    return df
