from __future__ import annotations

"""Price query helpers for raw long-format adjusted close time series."""

from typing import Optional, Sequence

import pandas as pd

from handyman.base import DateLike, build_security_filter_sql, run_sql_template
from sql.script_factory import default_sql_client
from utils.dataframe_utils import ensure_datetime_column


def get_prices(
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
) -> pd.DataFrame:
    """Return adjusted-close prices in long format.

    Parameters
    ----------
    tickers
        Optional ticker filter value(s).
    figis
        Optional FIGI filter value(s). Cannot be combined with ``tickers``.
    start_date
        Optional inclusive lower bound applied to ``DATE``.
    end_date
        Optional inclusive upper bound applied to ``DATE``.
    configs_path
        Optional path to the configuration file with Azure credentials.

    Returns
    -------
    pandas.DataFrame
        Long dataframe with required columns ``DATE``, ``TICKER``, and
        ``ADJ_CLOSE``.

    Raises
    ------
    ValueError
        If required columns ``DATE``, ``TICKER``, and ``ADJ_CLOSE`` are missing.
    """
    security_filter = build_security_filter_sql(
        sql_client=default_sql_client,
        tickers=tickers,
        figis=figis,
    )

    raw = run_sql_template(
        sql_file="prices.txt",
        filters={
            "security_filter": security_filter,
            "date_filter": "\n".join(
                filter_text
                for filter_text in [
                    default_sql_client.add_date_filter("DATE", start_date),
                    default_sql_client.add_end_date_filter("DATE", end_date),
                ]
                if filter_text
            ),
        },
        configs_path=configs_path,
    )
    raw = ensure_datetime_column(raw, "DATE")

    required_cols = {"DATE", "TICKER", "ADJ_CLOSE"}
    missing_cols = required_cols.difference(raw.columns)
    if missing_cols:
        raise ValueError(
            f"Expected columns {sorted(required_cols)}; missing {sorted(missing_cols)}"
        )

    return raw


def get_analyst_price_targets(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return analyst price target rows with optional ticker/date filtering.

    Parameters
    ----------
    tickers
        Optional ticker filter value(s).
    figis
        Optional FIGI filter value(s). Cannot be combined with ``tickers``.
    start_date
        Optional inclusive lower bound applied to ``DATE``.
    configs_path
        Optional path to the configuration file with Azure credentials.
    get_latest
        If True, return only rows from the latest snapshot date per ticker and
        ignore ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Analyst price target rows with ``DATE`` coerced to datetime.
    """
    security_filter = build_security_filter_sql(
        sql_client=default_sql_client,
        tickers=tickers,
        figis=figis,
    )
    if not get_latest:
        df = run_sql_template(
            sql_file="base_with_filters.txt",
            filters={
                "schema": "[dbo]",
                "table_name": "[analyst_price_targets]",
                "filters_sql": "\n".join(
                    filter_text
                    for filter_text in [
                        security_filter,
                        default_sql_client.add_date_filter("DATE", start_date),
                    ]
                    if filter_text
                ),
            },
            configs_path=configs_path,
        )
    else:
        df = run_sql_template(
            sql_file="analyst_price_targets_latest.txt",
            filters={
                "security_filter": security_filter,
                "date_filter": "",
            },
            configs_path=configs_path,
        )
    return ensure_datetime_column(df, "DATE")
