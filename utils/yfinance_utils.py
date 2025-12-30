import data_loader.yahoo_finance as yahoo_finance
import utils.dataframe_utils as dataframe_utils
import utils.azure_utils as azure_utils
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from utils.dataframe_utils import add_missing_tickers
from utils.list_utils import normalize_to_list


def create_client(tickers, max_workers: int = 10):
    """
    Create and return a YahooDataClient for one or more tickers.

    Parameters
    ----------
    tickers : str, sequence of str, or array-like
        Ticker symbol(s) to associate with the client.
    max_workers : int, default 10
        Maximum number of concurrent worker threads used by the client.

    Returns
    -------
    YahooDataClient
        Initialized Yahoo finance data client.

    Notes
    -----
    - `tickers` is normalized internally to support strings, lists, tuples,
      NumPy arrays, and pandas Series.
    - This function performs no I/O; it only constructs the client.
    """

    tickers = normalize_to_list(tickers)

    client = yahoo_finance.YahooDataClient(
        tickers,
        max_workers=max_workers,
    )

    return client


def pull_prices(tickers, client=None, configs_path=None):
    """
    Retrieve historical adjusted prices for a list of tickers starting from
    the next business day after the last date available in the database.

    The function ensures all requested tickers exist in the `prices` table,
    filling missing tickers with a default start date. It then uses a Yahoo
    Finance client to fetch prices from the database start date onward.

    Parameters
    ----------
    tickers : str, iterable of str, or array-like
        Tickers to fetch prices for. Strings will be wrapped in a list.
    client : YahooDataClient, optional
        Pre-initialized Yahoo Finance client. If None, a new client is created.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing historical prices for the requested tickers.

    Notes
    -----
    - Missing tickers are added with a default start date of '2000-01-01'.
    - The database query groups by ticker to determine the last date in the
      `prices` table for each ticker.
    """

    if not client:
        client = create_client(tickers=tickers)

    ticker_str = "', '".join(tickers)

    query = f"""
    SELECT TICKER, MAX(DATE) AS START_DATE
    FROM prices
    WHERE TICKER IN ('{ticker_str}')
    GROUP BY TICKER
    """

    engine = azure_utils.get_azure_engine(configs_path=configs_path)

    max_dates = azure_utils.read_sql_table(engine=engine, query=query)
    max_dates["START_DATE"] = pd.to_datetime(max_dates["START_DATE"])
    max_dates["START_DATE"] = max_dates["START_DATE"] + BDay(1)

    start_date_mapping = dataframe_utils.df_to_dict(
        add_missing_tickers(max_dates, client.tickers),
        "TICKER",
        "START_DATE",
    )

    all_prices = client.get_prices(start_date=start_date_mapping)
    all_prices["DATE"] = pd.to_datetime(all_prices["DATE"]).dt.date
    return all_prices


def pull_financials(
    tickers,
    annual: bool = True,
    statement_type: str = "financial",
    client=None,
):
    """
    Retrieve financial statements for one or more tickers.

    Parameters
    ----------
    tickers : str, sequence of str, or array-like
        Ticker symbol(s) to retrieve financials for.
    annual : bool, default True
        If True, return annual financial statements.
        If False, return quarterly statements.
    statement_type : {"financial", "balance_sheet", "income_statement", "cashflow"}
        Type of financial statement to retrieve.
    client : YahooDataClient, optional
        Pre-initialized Yahoo data client. If not provided, one will be created.

    Returns
    -------
    pandas.DataFrame
        Financial statement data for the requested tickers.

    Raises
    ------
    ValueError
        If `statement_type` is not one of the supported values.

    Notes
    -----
    - `tickers` is normalized internally to support strings, lists, tuples,
      NumPy arrays, and pandas Series.
    - Client creation is deferred to allow dependency injection in tests.
    """

    valid_statement_types = {
        "financial",
        "balance_sheet",
        "income_statement",
        "cashflow",
    }

    if statement_type not in valid_statement_types:
        raise ValueError(
            f"statement_type must be one of {valid_statement_types}, "
            f"got '{statement_type}'"
        )

    if not client:
        client = create_client(tickers=tickers)

    df = client.get_financials(
        annual=annual,
        statement_type=statement_type,
    )
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"]).dt.date

    return df


def pull_info(tickers, client=None):
    """
    Retrieve company information for one or more tickers.

    Parameters
    ----------
    tickers : str, sequence of str, or array-like
        Ticker symbol(s) to retrieve company information for.
        Ignored if `client` is provided.
    client : object, optional
        Pre-initialized data client with a `get_company_info()` method.
        If not provided, a new client is created using `create_client`.

    Returns
    -------
    pandas.DataFrame
        Company information with all values coerced to strings.

    Notes
    -----
    - All values are cast to string to ensure schema consistency
      (e.g., for database insertion or serialization).
    - Client injection enables deterministic unit testing.
    """

    if client is None:
        client = create_client(tickers=tickers)

    df = client.get_company_info()
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    df = df.map(lambda x: np.nan if isinstance(x, list) else x)

    return df


def pull_officers(tickers, client=None):
    """
    Retrieve company officer information for one or more tickers.

    Parameters
    ----------
    tickers : str, sequence of str, or array-like
        Ticker symbol(s) for which officer information should be retrieved.
        Ignored if `client` is provided.
    client : object, optional
        Pre-initialized data client with a `get_officer_info()` method.
        If not provided, a new client is created using `create_client`.

    Returns
    -------
    pandas.DataFrame
        Officer information for the requested tickers.

    Notes
    -----
    - Client injection enables deterministic unit testing.
    - No type coercion is performed; column dtypes are preserved.
    """

    if client is None:
        client = create_client(tickers=tickers)

    df = client.get_officer_info()
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    return df
