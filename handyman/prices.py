import pandas as pd
import os
import utils.azure_utils as azure_utils
from utils.list_utils import normalize_to_list
from utils.query_utils import render_sql_query


def get_prices(tickers=None, start_date=None, configs_path=None):
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

    ticker_filter = ""
    if tickers:
        ticker_string = "', '".join(tickers)
        ticker_filter = f"AND TICKER IN ('{ticker_string}')"

    query_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "sql",
            "prices.txt",
        )
    )

    query = render_sql_query(
        query_path=query_path,
        filters={"ticker_filter": ticker_filter, "date_filter": date_filter},
    )

    engine = azure_utils.get_azure_engine(configs_path=configs_path)

    df = azure_utils.read_sql_table(engine=engine, query=query)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.pivot(index="DATE", columns="TICKER", values="ADJ_CLOSE")
    df = df.ffill().where(df[::-1].notna().cummax()[::-1])
    return df
