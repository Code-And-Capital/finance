from __future__ import annotations

"""Company information query helpers."""

from typing import Optional, Sequence

import pandas as pd

from handyman.base import DateLike, read_table_by_filters, run_sql_template
from sql.script_factory import default_sql_client
from utils.dataframe_utils import ensure_datetime_column
from utils.list_utils import normalize_string_list


def get_company_info(
    tickers: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return company information with optional ticker/date filters.

    Parameters
    ----------
    tickers
        Optional ticker filter.
    start_date
        Optional inclusive lower-bound filter for ``DATE``.
    end_date
        Optional inclusive upper-bound filter for ``DATE``.
    configs_path
        Optional configs path for Azure credentials.
    get_latest
        If True, return only the latest row per ticker. In this mode
        ``start_date`` is ignored.

    Returns
    -------
    pandas.DataFrame
        Company information rows with ``DATE`` coerced to datetime.
    """
    normalized_tickers = normalize_string_list(tickers, field_name="tickers")
    ticker_filter = default_sql_client.add_in_filter("TICKER", normalized_tickers)
    date_filter = (
        ""
        if get_latest
        else "\n".join(
            part
            for part in [
                default_sql_client.add_date_filter("DATE", start_date),
                default_sql_client.add_end_date_filter("DATE", end_date),
            ]
            if part
        )
    )
    sql_file = "company_info_latest.txt" if get_latest else "company_info.txt"

    df = run_sql_template(
        sql_file=sql_file,
        filters={
            "ticker_filter": ticker_filter,
            "date_filter": date_filter,
        },
        configs_path=configs_path,
    )
    if "_ROW_NUM" in df.columns:
        df = df.drop(columns=["_ROW_NUM"])

    return ensure_datetime_column(df, "DATE")


def get_officers(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return officer rows with optional ticker/date filters.

    Parameters
    ----------
    tickers
        Optional ticker filter.
    start_date
        Optional inclusive lower-bound filter for ``DATE``.
    end_date
        Optional inclusive upper-bound filter for ``DATE``.
    configs_path
        Optional configs path for Azure credentials.
    get_latest
        If True, return only the latest officer snapshot per ticker and ignore
        ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Officer rows with ``DATE`` coerced to datetime.
    """
    if not get_latest:
        df = read_table_by_filters(
            table_name="officers",
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            configs_path=configs_path,
        )
        return ensure_datetime_column(df, "DATE")

    normalized_tickers = normalize_string_list(tickers, field_name="tickers")
    ticker_filter = default_sql_client.add_in_filter("TICKER", normalized_tickers)
    df = run_sql_template(
        sql_file="officers_latest.txt",
        filters={"ticker_filter": ticker_filter, "date_filter": ""},
        configs_path=configs_path,
    )
    return ensure_datetime_column(df, "DATE")
