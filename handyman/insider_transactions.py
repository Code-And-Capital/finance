from __future__ import annotations

"""Insider transaction query helpers."""

from typing import Optional, Sequence

import pandas as pd

from handyman.base import DateLike, read_table_by_filters, run_sql_template
from sql.script_factory import default_sql_client
from utils.dataframe_utils import ensure_datetime_column
from utils.list_utils import normalize_string_list


def get_insider_transactions(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return insider transaction rows with optional ticker/date filters.

    Parameters
    ----------
    tickers
        Optional ticker filter.
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
        Insider transaction rows with temporal columns coerced to datetime.
    """
    if not get_latest:
        df = read_table_by_filters(
            table_name="insider_transactions",
            tickers=tickers,
            start_date=start_date,
            configs_path=configs_path,
        )
    else:
        normalized_tickers = normalize_string_list(tickers, field_name="tickers")
        ticker_filter = default_sql_client.add_in_filter("TICKER", normalized_tickers)
        df = run_sql_template(
            sql_file="insider_transactions_latest.txt",
            filters={"ticker_filter": ticker_filter, "date_filter": ""},
            configs_path=configs_path,
        )
    for column in ["DATE", "START_DATE"]:
        if column in df.columns:
            df = ensure_datetime_column(df, column)
    return df
