import pandas as pd
import os
import utils.azure_utils as azure_utils
from utils.list_utils import normalize_to_list
from utils.query_utils import render_sql_query


def get_company_info(tickers=None, start_date=None, configs_path=None):
    """
    Retrieve company-level information with optional ticker and date filters.

    Parameters
    ----------
    tickers : str, sequence of str, or array-like, optional
        Ticker symbol(s) to filter on. A single string is treated as one ticker.
        If None, no ticker filtering is applied.
    start_date : str or datetime-like, optional
        Earliest date (inclusive) to include in the results.
        If None, no date filtering is applied.

    Returns
    -------
    pandas.DataFrame
        Company information data with the DATE column converted to datetime.

    Notes
    -----
    - Inputs are normalized using `normalize_to_list` to support strings,
      lists, tuples, NumPy arrays, and pandas Series.
    - SQL is constructed via string interpolation; inputs are assumed trusted.
    """

    tickers = normalize_to_list(tickers)

    ticker_filter = ""
    date_filter = ""

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
            "company_info.txt",
        )
    )

    query = render_sql_query(
        query_path=query_path,
        filters={"ticker_filter": ticker_filter, "date_filter": date_filter},
    )

    engine = azure_utils.get_azure_engine(configs_path=configs_path)

    df = azure_utils.read_sql_table(engine=engine, query=query)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df
