import pandas as pd
import utils.sql_utils as sql_utils
from utils.list_utils import normalize_to_list


def get_prices(tickers, start_date=None):
    """
    Retrieve adjusted close price history for a set of tickers and return it
    as a wide time-series DataFrame.

    This function queries the `prices` table in the CODE_CAPITAL database for
    adjusted close prices (`ADJ_CLOSE`) for the specified tickers on or after
    the given start date. The result is returned in pivoted (wide) format,
    indexed by date with one column per ticker.

    Parameters
    ----------
    tickers : Sequence[str]
        Collection of ticker symbols to query. Each ticker must exist in the
        `prices` table.
    start_date : str or datetime-like
        Earliest date (inclusive) for which prices should be retrieved.
        Passed directly into the SQL query.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by `DATE` (datetime64[ns]) with ticker symbols as
        columns and adjusted close prices as values.

    Notes
    -----
    - The SQL query is constructed via string interpolation; inputs are assumed
      to be trusted.
    - Missing prices for a given date-ticker combination will appear as NaN
      after pivoting.
    """
    tickers = normalize_to_list(tickers)

    date_filter = ""
    if start_date:
        date_filter = f"AND DATE >= '{start_date}'"

    sec_string = "', '".join(tickers)

    prices_query = f"""
    SELECT DATE, TICKER, ADJ_CLOSE
    FROM prices
    WHERE 1=1
    AND TICKER IN ('{sec_string}')
    {date_filter}
    """

    prices = sql_utils.read_sql_table(query=prices_query, database_name="CODE_CAPITAL")
    prices["DATE"] = pd.to_datetime(prices["DATE"])
    prices = prices.pivot(index="DATE", columns="TICKER", values="ADJ_CLOSE")
    return prices
