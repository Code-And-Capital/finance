from __future__ import annotations

"""Holdings query helpers for index and LLM strategy portfolios."""

from typing import Optional, Sequence

import pandas as pd

from handyman.base import DateLike, run_sql_template
from sql.script_factory import default_sql_client
from utils.dataframe_utils import ensure_datetime_column
from utils.list_utils import normalize_string_list


def get_index_holdings(
    indices: Optional[Sequence[str] | str] = None,
    tickers: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return index holdings filtered by index name, ticker, and date.

    Parameters
    ----------
    indices
        Optional index names for filtering (for example ``SP500``).
    tickers
        Optional ticker filter value(s).
    start_date
        Optional inclusive lower bound applied to ``DATE``.
    end_date
        Optional inclusive upper bound applied to ``DATE``.
    configs_path
        Optional path to the configuration file with Azure credentials.
    get_latest
        If True, return only the latest holdings snapshot per index and ignore
        ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Holdings rows with ``DATE`` coerced to datetime.
    """
    normalized_indices = normalize_string_list(indices, field_name="indices")
    normalized_tickers = normalize_string_list(tickers, field_name="tickers")
    sql_file = "holdings_latest.txt" if get_latest else "holdings.txt"
    date_filter = (
        ""
        if get_latest
        else "\n".join(
            filter_text
            for filter_text in [
                default_sql_client.add_date_filter("DATE", start_date),
                default_sql_client.add_end_date_filter("DATE", end_date),
            ]
            if filter_text
        )
    )

    df = run_sql_template(
        sql_file=sql_file,
        filters={
            "index_filter": default_sql_client.add_in_filter(
                '"INDEX"', normalized_indices
            ),
            "ticker_filter": default_sql_client.add_in_filter(
                "TICKER", normalized_tickers
            ),
            "date_filter": date_filter,
        },
        configs_path=configs_path,
    )

    return ensure_datetime_column(df, "DATE")


def get_llm_holdings(
    llms: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return LLM strategy holdings filtered by strategy and date.

    Parameters
    ----------
    llms
        Optional LLM strategy names for filtering.
    start_date
        Optional inclusive lower bound applied to ``DATE``.
    end_date
        Optional inclusive upper bound applied to ``DATE``.
    configs_path
        Optional path to the configuration file with Azure credentials.
    get_latest
        If True, return only the latest holdings snapshot per strategy and
        ignore ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Holdings rows with ``DATE`` coerced to datetime.
    """
    normalized_llms = normalize_string_list(llms, field_name="llms")
    sql_file = "llm_holdings_latest.txt" if get_latest else "llm_holdings.txt"
    date_filter = (
        ""
        if get_latest
        else "\n".join(
            filter_text
            for filter_text in [
                default_sql_client.add_date_filter("DATE", start_date),
                default_sql_client.add_end_date_filter("DATE", end_date),
            ]
            if filter_text
        )
    )

    df = run_sql_template(
        sql_file=sql_file,
        filters={
            "llm_filter": default_sql_client.add_in_filter("strategy", normalized_llms),
            "date_filter": date_filter,
        },
        configs_path=configs_path,
    )

    return ensure_datetime_column(df, "DATE")
