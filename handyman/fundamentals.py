from __future__ import annotations

"""Fundamentals query helpers."""

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

_FUNDAMENTAL_TABLES: dict[tuple[str, bool], str] = {
    ("financial", True): "financial_annual",
    ("financial", False): "financial_quarterly",
    ("balance_sheet", True): "balancesheet_annual",
    ("balance_sheet", False): "balancesheet_quarterly",
    ("income_statement", True): "incomestatement_annual",
    ("income_statement", False): "incomestatement_quarterly",
    ("cashflow", True): "cashflow_annual",
    ("cashflow", False): "cashflow_quarterly",
}

_ESTIMATES_TABLES: dict[str, str] = {
    "eps": "eps_estimates",
    "revenue": "revenue_estimates",
    "growth": "growth_estimates",
}


def get_fundamentals(
    *,
    statement_type: str,
    annual: bool,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return one fundamentals dataset by statement type and periodicity.

    Parameters
    ----------
    statement_type
        Statement family: ``financial``, ``balance_sheet``,
        ``income_statement``, or ``cashflow``.
    annual
        If True, query annual data; otherwise query quarterly data.
    tickers
        Optional ticker filter value(s).
    figis
        Optional FIGI filter value(s). Cannot be combined with ``tickers``.
    start_date
        Optional inclusive lower bound applied to the ``DATE`` column.
    configs_path
        Optional path to the configuration file with Azure credentials.
    get_latest
        If True, return the latest snapshot per ticker and ignore
        ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Fundamentals rows with ``DATE`` coerced to datetime.

    Raises
    ------
    ValueError
        If ``statement_type``/``annual`` does not map to a supported table.
    """
    key = (statement_type, annual)
    if key not in _FUNDAMENTAL_TABLES:
        allowed = ", ".join(
            f"{name}:{'annual' if is_annual else 'quarterly'}"
            for name, is_annual in _FUNDAMENTAL_TABLES
        )
        raise ValueError(
            f"Invalid fundamentals selection. Allowed combinations: {allowed}"
        )

    table_name = _FUNDAMENTAL_TABLES[key]
    if not get_latest:
        df = read_table_by_filters(
            table_name=table_name,
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
        sql_file="fundamentals_latest.txt",
        filters={
            "table_name": default_sql_client.quote_ident(table_name),
            "partition_column": default_sql_client.quote_ident("REPORT_DATE"),
            "security_filter": security_filter,
            "date_filter": "",
        },
        configs_path=configs_path,
    )
    return ensure_datetime_column(df, "DATE")


def get_eps_revisions(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return EPS revision rows with optional ticker/date filtering.

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
        If True, return the latest snapshot per ticker and ignore
        ``start_date``.

    Returns
    -------
    pandas.DataFrame
        EPS revision rows with ``DATE`` coerced to datetime.
    """
    if not get_latest:
        df = read_table_by_filters(
            table_name="eps_revisions",
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
            sql_file="eps_revisions_latest.txt",
            filters={"security_filter": security_filter, "date_filter": ""},
            configs_path=configs_path,
        )
    return ensure_datetime_column(df, "DATE")


def get_earnings_surprises(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return earnings surprise rows with optional ticker/date filtering.

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
        If True, return the latest snapshot per ticker and ignore
        ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Earnings surprise rows with date columns coerced to datetime.
    """
    if not get_latest:
        df = read_table_by_filters(
            table_name="earnings_surprises",
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
            sql_file="earnings_surprises_latest.txt",
            filters={"security_filter": security_filter, "date_filter": ""},
            configs_path=configs_path,
        )

    df = ensure_datetime_column(df, "DATE")
    if "EARNINGS_DATE" in df.columns:
        df = ensure_datetime_column(df, "EARNINGS_DATE")
    return df


def _get_estimates_by_type(
    *,
    estimate_type: str,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return one estimates dataset with optional ticker/date filtering."""
    table_name = _ESTIMATES_TABLES[estimate_type]
    if not get_latest:
        df = read_table_by_filters(
            table_name=table_name,
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
        sql_file="fundamentals_latest.txt",
        filters={
            "table_name": default_sql_client.quote_ident(table_name),
            "partition_column": default_sql_client.quote_ident("PERIOD"),
            "security_filter": security_filter,
            "date_filter": "",
        },
        configs_path=configs_path,
    )
    return ensure_datetime_column(df, "DATE")


def get_eps_estimates(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return EPS estimate rows with optional ticker/date filters.

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
        If True, return the latest snapshot per ticker and ignore
        ``start_date``.
    """
    return _get_estimates_by_type(
        estimate_type="eps",
        tickers=tickers,
        figis=figis,
        start_date=start_date,
        configs_path=configs_path,
        get_latest=get_latest,
    )


def get_revenue_estimates(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return revenue estimate rows with optional ticker/date filters.

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
        If True, return the latest snapshot per ticker and ignore
        ``start_date``.
    """
    return _get_estimates_by_type(
        estimate_type="revenue",
        tickers=tickers,
        figis=figis,
        start_date=start_date,
        configs_path=configs_path,
        get_latest=get_latest,
    )


def get_growth_estimates(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return growth estimate rows with optional ticker/date filters.

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
        If True, return the latest snapshot per ticker and ignore
        ``start_date``.
    """
    return _get_estimates_by_type(
        estimate_type="growth",
        tickers=tickers,
        figis=figis,
        start_date=start_date,
        configs_path=configs_path,
        get_latest=get_latest,
    )
