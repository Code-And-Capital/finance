from __future__ import annotations

"""Analyst recommendations query helpers."""

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


def get_analyst_recommendations(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return analyst recommendation rows with optional ticker/date filtering.

    Parameters
    ----------
    tickers
        Optional ticker filter value(s).
    figis
        Optional FIGI filter value(s). Cannot be combined with ``tickers``.
    start_date
        Optional inclusive lower bound applied to the ``DATE`` column.
    configs_path
        Optional path to the configuration file with Azure credentials.
    get_latest
        If True, return only the latest snapshot per ticker and ignore
        ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Analyst recommendation rows with ``DATE`` coerced to datetime.
    """
    if not get_latest:
        df = read_table_by_filters(
            table_name="analyst_recommendations",
            tickers=tickers,
            figis=figis,
            start_date=start_date,
            configs_path=configs_path,
        )
        return ensure_datetime_column(df, "DATE")

    security_filter = build_security_filter_sql(
        sql_client=default_sql_client,
        tickers=tickers,
        figis=figis,
    )
    df = run_sql_template(
        sql_file="analyst_recommendations_latest.txt",
        filters={"security_filter": security_filter, "date_filter": ""},
        configs_path=configs_path,
    )
    return ensure_datetime_column(df, "DATE")


def get_analyst_upgrades_downgrades(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return analyst upgrades/downgrades rows with optional filtering.

    Parameters
    ----------
    tickers
        Optional ticker filter value(s).
    figis
        Optional FIGI filter value(s). Cannot be combined with ``tickers``.
    start_date
        Optional inclusive lower bound applied to the ``DATE`` column.
    configs_path
        Optional path to the configuration file with Azure credentials.
    get_latest
        If True, return only the latest snapshot per ticker and ignore
        ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Analyst upgrades/downgrades rows with ``DATE`` and ``GRADEDATE``
        coerced to datetime when present.
    """
    if not get_latest:
        df = read_table_by_filters(
            table_name="analyst_upgrades_downgrades",
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
            sql_file="analyst_upgrades_downgrades_latest.txt",
            filters={"security_filter": security_filter, "date_filter": ""},
            configs_path=configs_path,
        )

    df = ensure_datetime_column(df, "DATE")
    if "GRADEDATE" in df.columns:
        df = ensure_datetime_column(df, "GRADEDATE")
    return df
