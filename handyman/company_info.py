import pandas as pd
import utils.sql_utils as sql_utils
from utils.list_utils import normalize_to_list


def get_company_info(tickers=None, start_date=None):
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

    query = f"""
    SELECT *
    FROM company_info
    WHERE 1=1
    {ticker_filter}
    {date_filter}
    """

    df = sql_utils.read_sql_table(query=query, database_name="CODE_CAPITAL")
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df
