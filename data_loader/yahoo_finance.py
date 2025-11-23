import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from copy import deepcopy
import time
from utils.threading import ThreadWorkerPool
import utils.logging as logging
import utils.dataframe_utils as dataframe_utils


class YahooDataClient:
    """
    Unified interface for retrieving Yahoo Finance data for multiple tickers,
    with built-in parallelization, retry handling, and dataframe normalization.

    This class provides high-level convenience methods for retrieving:
        - Company information (metadata from `.info`)
        - Officer information
        - Historical price data
        - Corporate actions (dividends, splits)
        - Financial statements (annual or quarterly)

    Parameters
    ----------
    tickers : list or np.ndarray
        List of Yahoo Finance ticker symbols.
    max_workers : int, default 8
        Maximum number of parallel threads to use.
    retries : int, default 3
        Number of retry attempts for failed Yahoo API calls.

    Notes
    -----
    Internally uses a `ThreadWorkerPool` to parallelize individual ticker fetches.
    """

    def __init__(self, tickers, max_workers=8, retries=3):
        """
        Initialize the YahooDataClient.

        Parameters
        ----------
        tickers : list or np.ndarray
            List of ticker symbols.
        max_workers : int
            Number of threads for parallel execution.
        retries : int
            Number of retries for failed fetch operations.
        """
        if isinstance(tickers, list) or isinstance(tickers, np.ndarray):
            self.tickers = list(tickers)
        else:
            raise ValueError("tickers must be a list or numpy array")

        self.yf_obj = yf.Tickers(" ".join(self.tickers))
        self.max_workers = max_workers
        self.pool = ThreadWorkerPool(max_workers=max_workers)
        self.retries = retries

    @staticmethod
    def add_metadata(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add standard metadata columns to a dataframe, ensuring they appear
        as the first two columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        ticker : str
            Ticker symbol to annotate with.

        Returns
        -------
        pd.DataFrame
            Dataframe with DATE and TICKER as the first two columns.
        """
        df["DATE"] = date.today()
        df["TICKER"] = ticker

        # Reorder columns so DATE and TICKER are first
        cols = ["DATE", "TICKER"] + [
            c for c in df.columns if c not in ("DATE", "TICKER")
        ]
        return df[cols]

    def _retry_fetch(self, func, ticker):
        """
        Retry wrapper for Yahoo API calls, logs the ticker if all retries fail.

        Parameters
        ----------
        func : callable
            Function to execute with retry logic.
        ticker : str
            Ticker symbol associated with this fetch.

        Returns
        -------
        Any
            Output of the wrapped function.

        Raises
        ------
        Exception
            Re-raises the last exception after exhausting retries.
        """
        for attempt in range(self.retries):
            try:
                return func()
            except Exception as e:
                if attempt < self.retries - 1:
                    time.sleep(0.7)
                else:
                    # Log ticker in error
                    logging.log(f"Failed to fetch data for {ticker}: {e}", type="error")
                    raise

    def _fetch_info(self, ticker, obj):
        """
        Internal thread worker: fetch summary .info metadata.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        obj : yfinance.Ticker
            Ticker object from yfinance.

        Returns
        -------
        pd.DataFrame or None
            One-row dataframe of cleaned info metadata.
        """

        def inner():
            info = deepcopy(obj.info)

            if info is None:
                raise NotImplementedError()
            info.pop("companyOfficers", None)
            df = pd.DataFrame([info])
            df = self.add_metadata(df, ticker)
            df = dataframe_utils.normalize_columns(df)
            return df

        return self._retry_fetch(inner, ticker)

    def _fetch_officers(self, ticker, obj):
        """
        Internal thread worker: fetch officer information from `.info`.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        obj : yfinance.Ticker
            Ticker object.

        Returns
        -------
        pd.DataFrame or None
            Officer data, or None if no officers available.
        """

        def inner():
            officers = obj.info.get("companyOfficers", [])
            if not officers:
                return None

            df = pd.DataFrame(officers)
            df = self.add_metadata(df, ticker)
            df = dataframe_utils.normalize_columns(df)
            return df

        return self._retry_fetch(inner, ticker)

    def _fetch_prices(self, ticker, obj, start_date):
        """
        Internal thread worker: fetch historical price data.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        obj : yfinance.Ticker
            Ticker object.
        start_date : str or datetime
            Start date for historical price retrieval.

        Returns
        -------
        pd.DataFrame or None
            Historical OHLCV data, or None if no data exists.
        """
        if isinstance(start_date, dict):
            start = start_date[ticker]
        else:
            start = start_date

        def inner():
            df = obj.history(start=start, auto_adjust=False).reset_index()
            if df.empty:
                return None
            df = dataframe_utils.normalize_columns(df)
            df["TICKER"] = ticker
            df["DATE"] = pd.to_datetime(df["DATE"], utc=True).dt.date
            return df

        return self._retry_fetch(inner, ticker)

    def _fetch_actions(self, ticker, obj):
        """
        Internal thread worker: fetch corporate actions
        such as dividends and splits.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        obj : yfinance.Ticker
            Ticker object.

        Returns
        -------
        pd.DataFrame or None
            Corporate actions dataframe.
        """

        def inner():
            df = obj.actions
            if df.empty:
                return None
            df = df.reset_index()
            df = dataframe_utils.normalize_columns(df)
            df["TICKER"] = ticker
            return df

        return self._retry_fetch(inner, ticker)

    def _fetch_financials(self, ticker, obj, statement_type, annual):
        """
        Internal thread worker: fetch financial statements.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        obj : yfinance.Ticker
            Ticker object.
        statement_type : {'financial', 'balance_sheet', 'income_statement', 'cashflow'}
            Type of financial statement to retrieve.
        annual : bool
            Whether to retrieve annual (True) or quarterly (False) data.

        Returns
        -------
        pd.DataFrame or None
            Transposed financial statement dataframe.
        """
        mapping = {
            ("financial", True): "financials",
            ("financial", False): "quarterly_financials",
            ("balance_sheet", True): "balance_sheet",
            ("balance_sheet", False): "quarterly_balance_sheet",
            ("income_statement", True): "income_stmt",
            ("income_statement", False): "quarterly_income_stmt",
            ("cashflow", True): "cash_flow",
            ("cashflow", False): "quarterly_cash_flow",
        }

        attr = mapping[(statement_type, annual)]

        def inner():
            df = getattr(obj, attr)
            if df is None or df.empty:
                return None

            df = df.T.reset_index(names="REPORT_DATE")
            df = self.add_metadata(df, ticker)
            df = dataframe_utils.normalize_columns(df)
            return df

        return self._retry_fetch(inner, ticker)

    def get_company_info(self):
        """
        Retrieve company metadata from the `.info` property for all tickers.

        Returns
        -------
        pd.DataFrame
            Combined dataframe of all tickers' metadata.
        """
        logging.log("Loading Company Information")
        tasks = [
            (lambda t=t, o=obj: self._fetch_info(t, o))
            for t, obj in self.yf_obj.tickers.items()
        ]
        results = self.pool.run(tasks)
        return pd.concat([r for r in results if r is not None], ignore_index=True)

    def get_officer_info(self):
        """
        Retrieve officer details for all tickers.

        Returns
        -------
        pd.DataFrame
            Combined officer information dataframe.
        """
        logging.log("Loading Officer Information")
        tasks = [
            (lambda t=t, o=obj: self._fetch_officers(t, o))
            for t, obj in self.yf_obj.tickers.items()
        ]
        results = self.pool.run(tasks)
        return pd.concat([r for r in results if r is not None], ignore_index=True)

    def get_prices(self, start_date="2000-01-01"):
        """
        Retrieve historical OHLCV price data for all tickers.

        Parameters
        ----------
        start_date : str, default '2000-01-01'
            Start date for historical price retrieval.

        Returns
        -------
        pd.DataFrame
            Combined price history dataframe.
        """
        logging.log("Loading Prices")
        tasks = [
            (lambda t=t, o=obj: self._fetch_prices(t, o, start_date))
            for t, obj in self.yf_obj.tickers.items()
        ]
        results = self.pool.run(tasks)
        return pd.concat([r for r in results if r is not None], ignore_index=True)

    def get_actions(self):
        """
        Retrieve dividends, splits, and other corporate actions.

        Returns
        -------
        pd.DataFrame
            Combined corporate actions dataframe.
        """
        logging.log("Loading Company Actions")
        tasks = [
            (lambda t=t, o=obj: self._fetch_actions(t, o))
            for t, obj in self.yf_obj.tickers.items()
        ]
        results = self.pool.run(tasks)
        return pd.concat([r for r in results if r is not None], ignore_index=True)

    def get_financials(self, statement_type="financial", annual=True):
        """
        Retrieve financial statement data for all tickers.

        Parameters
        ----------
        statement_type : {'financial', 'balance_sheet', 'income_statement', 'cashflow'}
            Type of financial statement to retrieve.
        annual : bool, default True
            Whether to retrieve annual statements. If False, retrieves quarterly.

        Returns
        -------
        pd.DataFrame
            Combined dataframe containing all requested financial statements.
        """
        s = statement_type.replace("_", " ").title()
        logging.log(f"Loading Company {s}")
        tasks = [
            (lambda t=t, o=obj: self._fetch_financials(t, o, statement_type, annual))
            for t, obj in self.yf_obj.tickers.items()
        ]
        results = self.pool.run(tasks)
        return pd.concat([r for r in results if r is not None], ignore_index=True)
