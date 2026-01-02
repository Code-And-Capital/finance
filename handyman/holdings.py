import pandas as pd
import os
import utils.azure_utils as azure_utils
from utils.list_utils import normalize_to_list
from utils.query_utils import render_sql_query


def get_index_holdings(indices=None, tickers=None, start_date=None, configs_path=None):
    """
    Retrieve index holdings from the holdings table with optional filters.

    Parameters
    ----------
    indices : str or sequence of str, optional
        Index identifier(s) to filter on (e.g. 'SP500' or ['SP500', 'NASDAQ100']).
    tickers : str or sequence of str, optional
        Ticker symbol(s) to filter on.
    start_date : str or datetime-like, optional
        Earliest date (inclusive) for which holdings should be returned.

    Returns
    -------
    pandas.DataFrame
        Holdings data with DATE converted to datetime.

    Notes
    -----
    - String inputs are treated as a single index or ticker.
    - SQL is constructed via string interpolation; inputs are assumed trusted.
    """

    indices = normalize_to_list(indices)
    tickers = normalize_to_list(tickers)

    index_filter = ""
    ticker_filter = ""
    date_filter = ""

    if indices:
        index_string = "', '".join(indices)
        index_filter = f"""AND "INDEX" IN ('{index_string}')"""

    if tickers:
        ticker_string = "', '".join(tickers)
        ticker_filter = f"AND TICKER IN ('{ticker_string}')"

    if start_date:
        date_filter = f"AND DATE >= '{start_date}'"

    query_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "sql",
            "holdings.txt",
        )
    )

    query = render_sql_query(
        query_path=query_path,
        filters={
            "ticker_filter": ticker_filter,
            "date_filter": date_filter,
            "index_filter": index_filter,
        },
    )

    engine = azure_utils.get_azure_engine(configs_path=configs_path)

    df = azure_utils.read_sql_table(engine=engine, query=query)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


def get_llm_holdings(llms=None, start_date=None, configs_path=None):
    """
    Retrieve LLM holdings from the database with optional filters for strategies
    and start date.

    Parameters
    ----------
    llms : str, sequence of str, or array-like, optional
        Strategy names to filter. Single string is treated as one strategy.
    start_date : str or datetime-like, optional
        Earliest date (inclusive) to filter the holdings.

    Returns
    -------
    pandas.DataFrame
        Holdings data for the requested strategies and date range.

    Notes
    -----
    - Uses `normalize_to_list` internally to handle single strings, sequences,
      or array-like inputs.
    - SQL query is constructed via string interpolation.
    """
    llms = normalize_to_list(llms)

    llm_filter = ""
    date_filter = ""

    if llms:
        llm_string = "', '".join(llms)
        llm_filter = f"""AND strategy IN ('{llm_string}')"""

    if start_date:
        date_filter = f"AND DATE >= '{start_date}'"

    query_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "sql",
            "llm_holdings.txt",
        )
    )

    query = render_sql_query(
        query_path=query_path,
        filters={"llm_filter": llm_filter, "date_filter": date_filter},
    )

    engine = azure_utils.get_azure_engine(configs_path=configs_path)

    df = azure_utils.read_sql_table(engine=engine, query=query)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df
