from __future__ import annotations

"""Yahoo Finance data-loading client.

This module provides a threaded client to fetch multiple Yahoo Finance datasets
for a list of tickers and return normalized pandas DataFrames.
"""

import time
import warnings
import logging
from copy import deepcopy
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yfinance as yf

import utils.dataframe_utils as dataframe_utils
from utils.logging import log
from utils.threading import TaskExecutionError, ThreadWorkerPool

TickerDateInput = Union[str, Dict[str, str]]
FetcherFn = Callable[[str, yf.Ticker], Optional[pd.DataFrame]]

_FINANCIAL_ATTRS: Dict[tuple[str, bool], str] = {
    ("financial", True): "financials",
    ("financial", False): "quarterly_financials",
    ("balance_sheet", True): "balance_sheet",
    ("balance_sheet", False): "quarterly_balance_sheet",
    ("income_statement", True): "income_stmt",
    ("income_statement", False): "quarterly_income_stmt",
    ("cashflow", True): "cash_flow",
    ("cashflow", False): "quarterly_cash_flow",
}

_ESTIMATE_ATTRS: Dict[str, str] = {
    "eps": "earnings_estimate",
    "revenue": "revenue_estimate",
    "growth": "growth_estimates",
}

_YF_LOG_FILTER_CONFIGURED = False


class _YahooNoiseFilter(logging.Filter):
    """Suppress known non-actionable yfinance 404 noise logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()
        return not (
            "no fundamentals data found for symbol" in message
            or ("http error 404" in message and "quotesummary" in message)
        )


def _configure_yfinance_logger() -> None:
    """Install a one-time log filter to suppress noisy yfinance 404 entries."""
    global _YF_LOG_FILTER_CONFIGURED
    if _YF_LOG_FILTER_CONFIGURED:
        return

    yf_logger = logging.getLogger("yfinance")
    # Suppress noisy package-level errors; pipeline-level logging handles outcomes.
    yf_logger.setLevel(logging.CRITICAL)
    yf_logger.addFilter(_YahooNoiseFilter())
    _YF_LOG_FILTER_CONFIGURED = True


class YahooDataClient:
    """Threaded Yahoo Finance client for multi-ticker data extraction.

    Parameters
    ----------
    tickers : sequence[str]
        Ticker symbols to query (for example, ``["AAPL", "MSFT"]``).
    max_workers : int, default 8
        Maximum number of worker threads used for parallel ticker fetches.
    retries : int, default 3
        Number of retry attempts for a failed ticker-level fetch.
    """

    def __init__(
        self,
        tickers: Sequence[str],
        max_workers: int = 8,
        retries: int = 3,
    ) -> None:
        _configure_yfinance_logger()

        if not isinstance(tickers, Sequence) or isinstance(tickers, (str, bytes)):
            raise ValueError("tickers must be a sequence of ticker strings")

        cleaned_tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if not cleaned_tickers:
            raise ValueError("tickers cannot be empty")

        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        if retries < 1:
            raise ValueError("retries must be >= 1")

        # Preserve input order while removing duplicates.
        self.tickers = list(dict.fromkeys(cleaned_tickers))
        self.yf_obj = yf.Tickers(" ".join(self.tickers))
        self.pool = ThreadWorkerPool(max_workers=max_workers)
        self.max_workers = max_workers
        self.retries = retries

    def _retry_fetch(
        self, func: Callable[[], Optional[pd.DataFrame]], ticker: str
    ) -> Optional[pd.DataFrame]:
        """Execute a fetch function with retry and consistent error logging."""
        for attempt in range(1, self.retries + 1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                if attempt < self.retries:
                    time.sleep(0.7)
                    continue
                raise

    def _iter_ticker_objects(
        self,
        tickers: Sequence[str] | None = None,
    ) -> list[tuple[str, yf.Ticker]]:
        """Return ticker-object pairs for all or a selected ticker subset."""
        if tickers is None:
            return list(self.yf_obj.tickers.items())

        selected: list[tuple[str, yf.Ticker]] = []
        for ticker in tickers:
            key = str(ticker).strip().upper()
            obj = self.yf_obj.tickers.get(key)
            if obj is None:
                # Fallback for subset retries: create missing ticker objects on demand
                # so retries cannot silently skip unresolved symbols.
                obj = yf.Ticker(key)
                self.yf_obj.tickers[key] = obj
            selected.append((key, obj))
        return selected

    def _run_parallel(
        self,
        fetcher: FetcherFn,
        log_message: str,
        tickers: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Run a ticker fetcher concurrently for all or selected tickers."""
        log(log_message)
        tasks = [
            (lambda t=ticker, o=obj: fetcher(t, o))
            for ticker, obj in self._iter_ticker_objects(tickers)
        ]
        try:
            results = self.pool.run(
                tasks,
                return_exceptions=True,
                stop_on_exception=True,
            )
        except TaskExecutionError as exc:
            partial_frames = [
                frame
                for frame in exc.partial_results
                if isinstance(frame, pd.DataFrame) and not frame.empty
            ]
            partial_df = self._concat_results(partial_frames)
            if self._is_rate_limit_error(exc.original_exception):
                rate_limit_error = RuntimeError(str(exc.original_exception))
                setattr(rate_limit_error, "partial_df", partial_df)
                raise rate_limit_error from exc.original_exception
            raise exc.original_exception from exc

        # Bubble up rate-limit failures so pipeline-level retry/backoff can run.
        for result in results:
            if isinstance(result, Exception) and self._is_rate_limit_error(result):
                raise result

        frame_results = [
            result for result in results if not isinstance(result, Exception)
        ]
        return self._concat_results(frame_results)

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Return True when an exception message indicates a retriable Yahoo failure."""
        message = str(exc).lower()
        retriable_markers = (
            "too many requests",
            "rate limited",
            "earnings date",
        )
        return any(marker in message for marker in retriable_markers)

    @staticmethod
    def _concat_results(results: Sequence[Optional[pd.DataFrame]]) -> pd.DataFrame:
        """Concatenate non-empty DataFrames and return an empty DataFrame when none exist."""
        frames = [frame for frame in results if frame is not None and not frame.empty]
        if not frames:
            return pd.DataFrame()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*"
                ),
                category=FutureWarning,
            )
            return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _add_metadata(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Insert ingestion metadata columns as leading fields."""
        out = df.copy()
        out["DATE"] = pd.Timestamp.now()
        out["TICKER"] = ticker
        cols = ["DATE", "TICKER"] + [
            c for c in out.columns if c not in ("DATE", "TICKER")
        ]
        return out[cols]

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns and enforce uniqueness after normalization."""
        out = dataframe_utils.normalize_columns(df)
        if out.columns.is_unique:
            return out

        deduped: list[str] = []
        seen: Dict[str, int] = {}
        for col in out.columns:
            count = seen.get(col, 0)
            seen[col] = count + 1
            deduped.append(col if count == 0 else f"{col}_{count + 1}")

        out.columns = deduped
        return out

    @staticmethod
    def _resolve_start_date(start_date: TickerDateInput, ticker: str) -> Optional[str]:
        """Resolve ``start_date`` for a ticker from a scalar date or ticker-date mapping."""
        if isinstance(start_date, dict):
            return start_date.get(ticker)
        return start_date

    def _fetch_info(self, ticker: str, obj: yf.Ticker) -> Optional[pd.DataFrame]:
        """Fetch top-level company info for one ticker."""

        def inner() -> Optional[pd.DataFrame]:
            info = deepcopy(obj.info)
            if not info:
                return None

            info.pop("companyOfficers", None)
            df = pd.DataFrame([info])
            # Normalize all missing-value representations consistently without
            # using replace/fillna paths that trigger pandas downcasting warnings.
            object_columns = df.select_dtypes(include=["object", "string"]).columns
            for column in object_columns:
                text = df[column].astype("string").str.strip().str.lower()
                missing_mask = text.isin({"nan", "none", ""})
                df.loc[missing_mask, column] = np.nan

            df = df.where(pd.notna(df), np.nan)
            return self._normalize(self._add_metadata(df, ticker))

        return self._retry_fetch(inner, ticker)

    def _fetch_officers(self, ticker: str, obj: yf.Ticker) -> Optional[pd.DataFrame]:
        """Fetch company officers extracted from ``Ticker.info``."""

        def inner() -> Optional[pd.DataFrame]:
            info = deepcopy(obj.info) or {}
            officers = info.get("companyOfficers", [])
            if not officers:
                return None

            df = pd.DataFrame(officers)
            return self._normalize(self._add_metadata(df, ticker))

        return self._retry_fetch(inner, ticker)

    def _fetch_prices(
        self,
        ticker: str,
        obj: yf.Ticker,
        start_date: TickerDateInput,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV prices for one ticker."""
        start = self._resolve_start_date(start_date, ticker)

        def inner() -> Optional[pd.DataFrame]:
            df = obj.history(start=start, auto_adjust=False).reset_index()
            if df.empty:
                return None
            out = df.copy()
            out["TICKER"] = ticker
            cols = ["TICKER"] + [column for column in out.columns if column != "TICKER"]
            return self._normalize(out[cols])

        return self._retry_fetch(inner, ticker)

    def _fetch_analyst_price_target(
        self, ticker: str, obj: yf.Ticker
    ) -> Optional[pd.DataFrame]:
        """Fetch analyst price target summary for one ticker."""

        def inner() -> Optional[pd.DataFrame]:
            raw = deepcopy(obj.analyst_price_targets)
            if not raw:
                return None

            df = pd.DataFrame([raw])
            return self._normalize(self._add_metadata(df, ticker))

        return self._retry_fetch(inner, ticker)

    def _fetch_financials(
        self,
        ticker: str,
        obj: yf.Ticker,
        statement_type: str,
        annual: bool,
    ) -> Optional[pd.DataFrame]:
        """Fetch one financial statement table for a ticker."""
        attr = _FINANCIAL_ATTRS.get((statement_type, annual))
        if attr is None:
            valid_types = sorted({key for key, _ in _FINANCIAL_ATTRS})
            raise ValueError(
                f"statement_type must be one of {valid_types}, got '{statement_type}'"
            )

        def inner() -> Optional[pd.DataFrame]:
            df = getattr(obj, attr)
            if df is None or df.empty:
                return None

            out = df.T.reset_index(names="REPORT_DATE")
            return self._normalize(self._add_metadata(out, ticker))

        return self._retry_fetch(inner, ticker)

    def _fetch_options(self, ticker: str, obj: yf.Ticker) -> Optional[pd.DataFrame]:
        """Fetch options chains (calls and puts) for all expirations of a ticker."""

        def inner() -> Optional[pd.DataFrame]:
            expirations = list(obj.options or [])
            if not expirations:
                return None

            frames: list[pd.DataFrame] = []
            for expiration in expirations:
                chain = obj.option_chain(expiration)

                calls = chain.calls.copy()
                calls["option_type"] = "Call"
                calls["expiration"] = pd.to_datetime(expiration)

                puts = chain.puts.copy()
                puts["option_type"] = "Put"
                puts["expiration"] = pd.to_datetime(expiration)

                frames.extend([calls, puts])

            if not frames:
                return None

            df = pd.concat(frames, ignore_index=True)
            return self._normalize(self._add_metadata(df, ticker))

        return self._retry_fetch(inner, ticker)

    def _fetch_insider_transactions(
        self, ticker: str, obj: yf.Ticker
    ) -> Optional[pd.DataFrame]:
        """Fetch insider transactions and map free-text descriptions to class labels."""

        def inner() -> Optional[pd.DataFrame]:
            raw = deepcopy(obj.insider_transactions)
            if raw is None or raw.empty or "Text" not in raw.columns:
                return None

            df = raw.copy()
            text_series = df["Text"].fillna("").astype(str)

            df["Transaction"] = "BUY"
            df.loc[
                text_series.str.contains("sale", case=False, regex=False), "Transaction"
            ] = "SELL"
            df.loc[
                text_series.str.contains("gift", case=False, regex=False), "Transaction"
            ] = "GIFT"
            df.loc[
                text_series.str.contains("grant", case=False, regex=False),
                "Transaction",
            ] = "GRANT"
            df.loc[
                text_series.str.contains("conversion", case=False, regex=False),
                "Transaction",
            ] = "CONVERSION"
            df.loc[
                text_series.str.contains("purchase", case=False, regex=False),
                "Transaction",
            ] = "BUY"

            return self._normalize(self._add_metadata(df, ticker))

        return self._retry_fetch(inner, ticker)

    def _fetch_analyst_estimate(
        self,
        ticker: str,
        obj: yf.Ticker,
        estimate_type: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch one analyst estimate dataset for a ticker."""
        attr = _ESTIMATE_ATTRS.get(estimate_type)
        if attr is None:
            raise ValueError(
                f"estimate_type must be one of {sorted(_ESTIMATE_ATTRS)}, got '{estimate_type}'"
            )

        def inner() -> Optional[pd.DataFrame]:
            raw = getattr(obj, attr, None)
            if raw is None or raw.empty:
                return None

            df = raw.reset_index().copy()
            df["ESTIMATE_TYPE"] = estimate_type.upper()
            return self._normalize(self._add_metadata(df, ticker))

        return self._retry_fetch(inner, ticker)

    def _fetch_table_attr(
        self,
        ticker: str,
        obj: yf.Ticker,
        attr: str,
        reset_index: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Fetch a DataFrame-like ticker attribute and standardize output shape."""

        def inner() -> Optional[pd.DataFrame]:
            raw = deepcopy(getattr(obj, attr, None))
            if raw is None:
                return None

            if isinstance(raw, pd.DataFrame):
                df = raw
            else:
                df = pd.DataFrame(raw)

            if df.empty:
                return None

            out = df.reset_index() if reset_index else df
            return self._normalize(self._add_metadata(out, ticker))

        return self._retry_fetch(inner, ticker)

    def get_company_info(self) -> pd.DataFrame:
        """Return normalized company profile info for all configured tickers."""
        return self._run_parallel(self._fetch_info, "Loading Company Information")

    def get_officer_info(self) -> pd.DataFrame:
        """Return normalized company officer records for all configured tickers."""
        return self._run_parallel(self._fetch_officers, "Loading Officer Information")

    def get_prices(self, start_date: TickerDateInput = "2000-01-01") -> pd.DataFrame:
        """Return historical prices for all tickers.

        Parameters
        ----------
        start_date : str or dict[str, str], default "2000-01-01"
            Either a single start date for all tickers or a per-ticker mapping.
        """
        return self._run_parallel(
            lambda ticker, obj: self._fetch_prices(ticker, obj, start_date),
            "Loading Prices",
        )

    def get_analyst_price_targets(self) -> pd.DataFrame:
        """Return analyst price target snapshots for all configured tickers."""
        return self._run_parallel(
            self._fetch_analyst_price_target,
            "Loading Analyst Price Targets",
        )

    def get_actions(self) -> pd.DataFrame:
        """Return splits/dividends/corporate actions for all configured tickers."""
        return self._run_parallel(
            lambda ticker, obj: self._fetch_table_attr(ticker, obj, "actions"),
            "Loading Company Actions",
        )

    def get_options(self) -> pd.DataFrame:
        """Return options chains (calls and puts) for each available expiration date."""
        return self._run_parallel(self._fetch_options, "Loading Options")

    def get_financials(
        self, statement_type: str = "financial", annual: bool = True
    ) -> pd.DataFrame:
        """Return annual or quarterly financial statements across all tickers.

        Parameters
        ----------
        statement_type : {"financial", "balance_sheet", "income_statement", "cashflow"}, default "financial"
            Statement family to fetch.
        annual : bool, default True
            ``True`` for annual statements, ``False`` for quarterly.
        """
        statement_label = statement_type.replace("_", " ").title()
        period_label = "Annual" if annual else "Quarterly"
        return self._run_parallel(
            lambda ticker, obj: self._fetch_financials(
                ticker, obj, statement_type, annual
            ),
            f"Loading Company {period_label} {statement_label}",
        )

    def get_analyst_estimates(self, estimate_type: str) -> pd.DataFrame:
        """Return analyst estimate tables for all tickers.

        Parameters
        ----------
        estimate_type : {"eps", "revenue", "growth"}
            Estimate dataset selector.
        """
        if estimate_type not in _ESTIMATE_ATTRS:
            raise ValueError(
                f"estimate_type must be one of {sorted(_ESTIMATE_ATTRS)}, got '{estimate_type}'"
            )
        return self._run_parallel(
            lambda ticker, obj: self._fetch_analyst_estimate(
                ticker, obj, estimate_type
            ),
            f"Loading Analyst Estimates ({estimate_type.upper()})",
        )

    def get_recommendations(self) -> pd.DataFrame:
        """Return analyst recommendations for all configured tickers."""
        return self._run_parallel(
            lambda ticker, obj: self._fetch_table_attr(ticker, obj, "recommendations"),
            "Loading Analyst Recommendations",
        )

    def get_upgrades_downgrades(self) -> pd.DataFrame:
        """Return analyst upgrades and downgrades for all configured tickers."""
        return self._run_parallel(
            lambda ticker, obj: self._fetch_table_attr(
                ticker, obj, "upgrades_downgrades"
            ),
            "Loading Analyst Upgrades & Downgrades",
        )

    def get_eps_revisions(self) -> pd.DataFrame:
        """Return EPS revisions for all configured tickers."""
        return self._run_parallel(
            lambda ticker, obj: self._fetch_table_attr(ticker, obj, "eps_revisions"),
            "Loading EPS Revisions",
        )

    def get_earnings_surprises(self) -> pd.DataFrame:
        """Return earnings date/surprise records for all configured tickers."""
        return self._run_parallel(
            lambda ticker, obj: self._fetch_table_attr(ticker, obj, "earnings_dates"),
            "Loading Earnings Surprises",
        )

    def get_institutional_holders(self) -> pd.DataFrame:
        """Return institutional holder data for all configured tickers."""
        return self._run_parallel(
            lambda ticker, obj: self._fetch_table_attr(
                ticker,
                obj,
                "institutional_holders",
                reset_index=False,
            ),
            "Loading Institutional Holders",
        )

    def get_major_holders(self) -> pd.DataFrame:
        """Return major holder summaries for all configured tickers."""
        return self._run_parallel(
            lambda ticker, obj: self._fetch_table_attr(ticker, obj, "major_holders"),
            "Loading Major Holders",
        )

    def get_insider_transactions(self) -> pd.DataFrame:
        """Return insider transactions with normalized transaction class labels.

        Classification is derived from the text description and mapped to one of:
        ``SELL``, ``GIFT``, ``GRANT``, ``CONVERSION``, or ``BUY``.
        """
        return self._run_parallel(
            self._fetch_insider_transactions, "Loading Insider Transactions"
        )
