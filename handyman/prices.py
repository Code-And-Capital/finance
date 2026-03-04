from __future__ import annotations

"""Price query helpers for raw long-format adjusted close time series."""

from typing import Optional, Sequence

import pandas as pd

from handyman.base import DateLike, run_sql_template
from sql.script_factory import default_sql_client
from utils.dataframe_utils import ensure_datetime_column
from utils.list_utils import normalize_string_list


def get_prices(
    tickers: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
) -> pd.DataFrame:
    """Return adjusted-close prices in long format.

    Parameters
    ----------
    tickers
        Optional ticker filter value(s).
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
    normalized_tickers = normalize_string_list(tickers, field_name="tickers")

    raw = run_sql_template(
        sql_file="prices.txt",
        filters={
            "ticker_filter": default_sql_client.add_in_filter(
                "TICKER", normalized_tickers
            ),
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
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return analyst price target rows with optional ticker/date filtering.

    Parameters
    ----------
    tickers
        Optional ticker filter value(s).
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
    normalized_tickers = normalize_string_list(tickers, field_name="tickers")
    if not get_latest:
        df = run_sql_template(
            sql_file="base_with_filters.txt",
            filters={
                "schema": "[dbo]",
                "table_name": "[analyst_price_targets]",
                "filters_sql": "\n".join(
                    filter_text
                    for filter_text in [
                        default_sql_client.add_in_filter("TICKER", normalized_tickers),
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
                "ticker_filter": default_sql_client.add_in_filter(
                    "TICKER", normalized_tickers
                ),
                "date_filter": "",
            },
            configs_path=configs_path,
        )
    return ensure_datetime_column(df, "DATE")
