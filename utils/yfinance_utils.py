import data_loader.yahoo_finance as yahoo_finance
import utils.dataframe_utils as dataframe_utils
import utils.azure_utils as azure_utils
import pandas as pd
import numpy as np
import os
from pandas.tseries.offsets import BDay
from utils.dataframe_utils import add_missing_tickers
from utils.list_utils import normalize_to_list
from utils.query_utils import render_sql_query


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


def pull_prices(tickers=None, client=None, configs_path=None):
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

    ticker_filter = ""
    tickers = normalize_to_list(tickers)
    if tickers:
        ticker_string = "', '".join(tickers)
        ticker_filter = f"AND TICKER IN ('{ticker_string}')"

    query_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "sql",
            "max_date.txt",
        )
    )

    query = render_sql_query(
        query_path=query_path,
        filters={"ticker_filter": ticker_filter, "table_name": "prices"},
    )

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

    def _coerce_company_info_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize a company_info DataFrame to match the SQL schema:
        - Convert epoch/strings to date for known date columns.
        - Cast large integer-like columns to pandas nullable Int64.
        - Cast ratio/price columns to float.
        Call this right before write_sql_table(..., overwrite=False) to avoid
        data-type binding errors on append.
        """
        df = df.copy()

        date_cols = [
            "DATE",
            "LASTFISCALYEAREND",
            "NEXTFISCALYEAREND",
            "MOSTRECENTQUARTER",
            "EXDIVIDENDDATE",
            "LASTDIVIDENDDATE",
            "DIVIDENDDATE",
        ]

        bigint_cols = [
            "MARKETCAP",
            "ENTERPRISEVALUE",
            "TOTALCASH",
            "EBITDA",
            "TOTALDEBT",
            "TOTALREVENUE",
            "GROSSPROFITS",
            "OPERATINGCASHFLOW",
            "FLOATSHARES",
            "SHARESOUTSTANDING",
            "IMPLIEDSHARESOUTSTANDING",
            "FIRSTTRADEDATEMILLISECONDS",
            "DATESHORTINTEREST",
            "SHARESSHORTPRIORMONTH",
        ]

        float_cols = [
            "PAYOUTRATIO",
            "TRAILINGPE",
            "FORWARDPE",
            "PRICETOSALESTRAILING12MONTHS",
            "TRAILINGANNUALDIVIDENDYIELD",
            "PROFITMARGINS",
            "PRICETOBOOK",
            "NETINCOMETOCOMMON",
            "TARGETMEANPRICE",
            "52WEEKCHANGE",
            "SANDP52WEEKCHANGE",
            "TRAILINGPEGRATIO",
            "RETURNONEQUITY",
            "GMTOFFSETMILLISECONDS",
            "POSTMARKETTIME",
            "REGULARMARKETTIME",
            "PRICEEPSCURRENTYEAR",
            "FIFTYDAYAVERAGECHANGE",
            "FIFTYDAYAVERAGECHANGEPERCENT",
            "TWOHUNDREDDAYAVERAGECHANGE",
            "TWOHUNDREDDAYAVERAGECHANGEPERCENT",
            "POSTMARKETCHANGEPERCENT",
            "POSTMARKETCHANGE",
            "FIFTYTWOWEEKLOWCHANGE",
            "FIFTYTWOWEEKLOWCHANGEPERCENT",
            "FIFTYTWOWEEKHIGHCHANGEPERCENT",
            "FIFTYTWOWEEKCHANGEPERCENT",
            "GOVERNANCEEPOCHDATE",
        ]

        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.date

        for col in bigint_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    if client is None:
        client = create_client(tickers=tickers)

    df = client.get_company_info()
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    df = df.map(lambda x: np.nan if isinstance(x, list) else x)
    df = _coerce_company_info_types(df)
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
