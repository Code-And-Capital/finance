from __future__ import annotations

"""Shared query helpers for handyman data access."""

from datetime import date, datetime
from typing import Optional, Sequence, Union

import pandas as pd

from connectors.azure_data_source import default_azure_data_source
from sql.script_factory import SQLClient
from utils.list_utils import normalize_string_list

DateLike = Union[str, date, datetime, pd.Timestamp]


def run_sql_template(
    *, sql_file: str, filters: dict[str, str], configs_path: Optional[str]
) -> pd.DataFrame:
    """Render and execute a SQL template against Azure SQL.

    Parameters
    ----------
    sql_file
        Template file name under ``sql/templates``.
    filters
        Placeholder replacement mapping used by ``SQLClient.render_sql_query``.
    configs_path
        Optional path to the configuration file with Azure credentials.

    Returns
    -------
    pandas.DataFrame
        Query result rows.
    """
    sql = SQLClient()
    query = sql.render_sql_query(
        query_path=sql.resolve_sql_path(sql_file),
        filters=filters,
    )
    engine = default_azure_data_source.get_engine(configs_path=configs_path)
    return default_azure_data_source.read_sql_table(query=query, engine=engine)


def read_table_by_filters(
    *,
    table_name: str,
    tickers: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    ticker_column: str = "TICKER",
    date_column: str = "DATE",
) -> pd.DataFrame:
    """Read an Azure table using optional ticker and date filters.

    Parameters
    ----------
    table_name
        Target Azure SQL table name.
    tickers
        Optional ticker filter value(s).
    start_date
        Optional inclusive lower bound applied to ``date_column``.
    end_date
        Optional inclusive upper bound applied to ``date_column``.
    configs_path
        Optional path to the configuration file with Azure credentials.
    ticker_column
        Column name used for ticker filtering.
    date_column
        Column name used for date filtering.

    Returns
    -------
    pandas.DataFrame
        Query result rows from the selected table.
    """
    sql_client = SQLClient()
    normalized_tickers = normalize_string_list(tickers, field_name="tickers")
    ticker_filter = sql_client.add_in_filter(
        sql_client.quote_ident(ticker_column),
        normalized_tickers,
    )
    date_filter = sql_client.add_date_filter(
        f"CAST({sql_client.quote_ident(date_column)} AS date)",
        start_date,
    )
    end_date_filter = sql_client.add_end_date_filter(
        f"CAST({sql_client.quote_ident(date_column)} AS date)",
        end_date,
    )
    query = sql_client.build_select_with_filters_query(
        table_name=table_name,
        filters_sql=f"{ticker_filter}\n{date_filter}\n{end_date_filter}".rstrip(),
    )
    engine = default_azure_data_source.get_engine(configs_path=configs_path)
    return default_azure_data_source.read_sql_table(query=query, engine=engine)
