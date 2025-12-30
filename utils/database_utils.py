import utils.azure_utils as azure_utils
from utils.list_utils import normalize_to_list


def find_missing_tickers(table_name, tickers, configs_path=None):
    """
    Identify which tickers from a provided input are missing in a database table.

    The function queries the database for all distinct tickers appearing since
    the most recent date where more than 10 tickers were present. It then
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

    query = f"""
        WITH cutoff_date AS (
            SELECT MAX([DATE]) AS max_date
            FROM (
                SELECT [DATE]
                FROM {table_name}
                GROUP BY [DATE]
                HAVING COUNT(*) > 50
            ) d
        )
        SELECT DISTINCT ci.TICKER
        FROM {table_name} ci
        CROSS JOIN cutoff_date cd
        WHERE ci.[DATE] >= cd.max_date;
    """

    engine = azure_utils.get_azure_engine(configs_path=configs_path)

    df = azure_utils.read_sql_table(engine=engine, query=query)

    # Find tickers that are missing from the dataframe
    existing = set(df["TICKER"])
    missing = [t for t in tickers if t not in existing]

    return missing
