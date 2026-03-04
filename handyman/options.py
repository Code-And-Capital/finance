from __future__ import annotations

"""Options query helpers."""

from typing import Optional, Sequence

import pandas as pd

from handyman.base import DateLike, read_table_by_filters, run_sql_template
from sql.script_factory import default_sql_client
from utils.dataframe_utils import ensure_datetime_column
from utils.list_utils import normalize_string_list


def get_options(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return options rows with optional ticker/date filtering.

    Parameters
    ----------
    tickers
        Optional ticker filter.
    start_date
        Optional inclusive lower-bound filter for ``DATE``.
    configs_path
        Optional configs path for Azure credentials.
    get_latest
        If True, return only the latest options snapshot per
        ``TICKER``/``CONTRACTSYMBOL`` and ignore ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Options rows with temporal columns coerced to datetime where present.
    """
    if not get_latest:
        df = read_table_by_filters(
            table_name="options",
            tickers=tickers,
            start_date=start_date,
            configs_path=configs_path,
        )
    else:
        normalized_tickers = normalize_string_list(tickers, field_name="tickers")
        ticker_filter = default_sql_client.add_in_filter("TICKER", normalized_tickers)
        df = run_sql_template(
            sql_file="options_latest.txt",
            filters={"ticker_filter": ticker_filter, "date_filter": ""},
            configs_path=configs_path,
        )
        if "_ROW_NUM" in df.columns:
            df = df.drop(columns=["_ROW_NUM"])

    for column in ["DATE", "LASTTRADEDATE", "EXPIRATION"]:
        if column in df.columns:
            df = ensure_datetime_column(df, column)

    return df
