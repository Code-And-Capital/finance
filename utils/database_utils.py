import os
import utils.azure_utils as azure_utils
from utils.list_utils import normalize_to_list
from utils.query_utils import render_sql_query


def find_missing_tickers(table_name, tickers, configs_path=None):
    """
    Identify which tickers from a provided input are missing in a database table.

    The function queries the database for all distinct tickers appearing since
    the most recent date where more than 50 tickers were present. It then
    returns the subset of `tickers` that are not found in the database.

    Parameters
    ----------
    table_name : str
        Name of the database table to query.
    tickers : str, sequence of str, or array-like
        Ticker(s) to check. Strings are treated as single tickers; sequences
        or arrays are converted to lists.

    Returns
    -------
    list of str
        Tickers from `tickers` that are missing in the database.

    Notes
    -----
    - Uses `normalize_to_list` internally to handle single strings, lists, tuples,
      or numpy/pandas array-like objects.
    """

    tickers = normalize_to_list(tickers)
    ticker_string = "', '".join(tickers)
    ticker_filter = f"AND TICKER IN ('{ticker_string}')"

    query_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "sql",
            "missing_tickers.txt",
        )
    )

    query = render_sql_query(
        query_path=query_path,
        filters={"ticker_filter": ticker_filter, "table_name": table_name},
    )

    engine = azure_utils.get_azure_engine(configs_path=configs_path)

    df = azure_utils.read_sql_table(engine=engine, query=query)

    # Find tickers that are missing from the dataframe
    existing = set(df["TICKER"])
    missing = [t for t in tickers if t not in existing]

    return missing
