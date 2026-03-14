"""Shared Yahoo-based pipeline primitives."""

import time
from datetime import date as dt_date, datetime, timedelta
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from connectors.azure_data_source import default_azure_data_source
import connectors.yahoo_data_source as yahoo_data_source
from sql.script_factory import SQLClient
import utils.dataframe_utils as dataframe_utils
from utils.list_utils import normalize_to_list
from utils.logging import log


class YahooData:
    """Base class for Yahoo-driven pipeline components."""

    _RATE_LIMIT_WAIT_SECONDS = 120
    _MISSING_TICKER_WAIT_SECONDS = 120
    _MAX_MISSING_TICKER_RESETS = 10

    def __init__(
        self,
        *,
        tickers: Iterable[str] | str | None = None,
        yahoo_client=None,
        client=None,
        max_workers: int = 10,
    ) -> None:
        """Initialize common Yahoo pipeline state.

        Parameters
        ----------
        tickers
            Ticker symbols for Yahoo pulls.
        yahoo_client
            Explicit Yahoo client instance.
        client
            Backward-compatible alias for ``yahoo_client``.
        max_workers
            Maximum worker threads for created Yahoo clients.
        """
        if yahoo_client is not None and client is not None:
            raise ValueError("Pass only one of 'yahoo_client' or 'client'.")

        self.tickers = normalize_to_list(tickers)
        self.yahoo_client = yahoo_client if yahoo_client is not None else client
        self.max_workers = max_workers
        self.clients_used: list[object] = []
        self._sleep = time.sleep
        self.azure_data_source = default_azure_data_source
        self.sql_client = SQLClient()

    def _track_client(self, client) -> None:
        """Track a client instance once for lifecycle introspection."""
        if not any(existing is client for existing in self.clients_used):
            self.clients_used.append(client)

    def _create_client_for_tickers(self, tickers: Iterable[str] | str | None):
        """Create a YahooDataClient for an arbitrary ticker set and track it."""
        client = yahoo_data_source.YahooDataClient(
            normalize_to_list(tickers),
            max_workers=self.max_workers,
        )
        self._track_client(client)
        return client

    def _create_client(self):
        """Create a YahooDataClient for configured tickers."""
        return self._create_client_for_tickers(self.tickers)

    def _resolve_client(self):
        """Resolve an injected Yahoo client or create a default one."""
        if self.yahoo_client is not None:
            self._track_client(self.yahoo_client)
            return self.yahoo_client
        self.yahoo_client = self._create_client()
        return self.yahoo_client

    def _pull_generic(
        self,
        client_method: str,
        method_kwargs: dict[str, Any] | None = None,
        date_columns: list[str] | None = None,
        handle_rate_limit: bool = True,
    ) -> pd.DataFrame:
        """Invoke a Yahoo client method and normalize selected date columns.

        Only date columns present in the returned dataframe are coerced.
        This keeps the helper resilient when upstream fields are optional.
        """
        client = self._resolve_client()
        method = getattr(client, client_method)
        kwargs = method_kwargs or {}
        while True:
            try:
                dataframe = method(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                if not handle_rate_limit:
                    raise
                if not self._is_rate_limit_error(exc):
                    raise
                log(
                    f"Rate limit while calling Yahoo method '{client_method}'. "
                    f"Waiting {self._RATE_LIMIT_WAIT_SECONDS} seconds before retry.",
                    type="warning",
                )
                self._sleep(self._RATE_LIMIT_WAIT_SECONDS)

        out = dataframe.copy()
        for column in date_columns or []:
            if column in out.columns:
                out = dataframe_utils.ensure_datetime_column(out, column)
        return out

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
    def _normalize_ticker_list(tickers: Iterable[str] | str | None) -> list[str]:
        """Normalize ticker symbols to uppercase strings with stable order."""
        if tickers is None:
            return []
        if isinstance(tickers, str):
            values = [tickers]
        else:
            values = list(tickers)
        return list(
            dict.fromkeys(
                str(ticker).strip().upper() for ticker in values if str(ticker).strip()
            )
        )

    @staticmethod
    def _normalize_filter_values(values: Iterable[str] | str | None) -> list[str]:
        """Normalize generic SQL filter values to distinct non-empty strings."""
        if values is None:
            return []
        if isinstance(values, str):
            raw = [values]
        else:
            raw = list(values)
        return list(
            dict.fromkeys(str(value).strip() for value in raw if str(value).strip())
        )

    def _extract_figi_values(self, dataframe: pd.DataFrame) -> list[str]:
        """Extract normalized FIGI values from a dataframe."""
        if dataframe.empty or "FIGI" not in dataframe.columns:
            return []
        return self._normalize_filter_values(
            dataframe["FIGI"].dropna().astype(str).tolist()
        )

    @staticmethod
    def _extract_returned_tickers(dataframe: pd.DataFrame) -> set[str]:
        """Extract returned ticker symbols from a DataFrame pull result."""
        if dataframe.empty or "TICKER" not in dataframe.columns:
            return set()
        return set(dataframe["TICKER"].dropna().astype(str).str.upper())

    def _pull_with_missing_ticker_retries(
        self,
        *,
        client_method: str,
        method_kwargs: dict[str, Any] | None = None,
        date_columns: list[str] | None = None,
        max_resets: int | None = None,
        wait_seconds: int | None = None,
    ) -> pd.DataFrame:
        """Retry pull calls for missing tickers until complete or reset limit is reached."""
        pending_tickers = self._normalize_ticker_list(self.tickers)
        total_tickers = len(pending_tickers)
        if total_tickers == 0:
            return pd.DataFrame()

        reset_limit = (
            self._MAX_MISSING_TICKER_RESETS if max_resets is None else max_resets
        )
        wait = (
            self._MISSING_TICKER_WAIT_SECONDS if wait_seconds is None else wait_seconds
        )
        reset_attempts = 0
        consecutive_empty_attempts = 0
        frames: list[pd.DataFrame] = []
        original_client = self.yahoo_client

        while pending_tickers:
            self.yahoo_client = self._create_client_for_tickers(pending_tickers)
            try:
                try:
                    attempt_df = self._pull_generic(
                        client_method=client_method,
                        method_kwargs=method_kwargs or {},
                        date_columns=date_columns,
                        handle_rate_limit=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    if not self._is_rate_limit_error(exc):
                        raise

                    partial_df = getattr(exc, "partial_df", pd.DataFrame())
                    if not partial_df.empty:
                        frames.append(partial_df)
                        completed_from_partial = self._extract_returned_tickers(
                            partial_df
                        )
                        pending_tickers = [
                            ticker
                            for ticker in pending_tickers
                            if ticker not in completed_from_partial
                        ]
                        log(
                            f"Rate limit hit for '{client_method}'. "
                            f"Stored partial rows: {len(partial_df)}; "
                            f"{len(pending_tickers)} tickers remaining."
                        )
                    else:
                        log(
                            f"Rate limit hit for '{client_method}' with no partial rows. "
                            f"{len(pending_tickers)} tickers still pending.",
                            type="warning",
                        )

                    if not pending_tickers:
                        break
                    if reset_attempts >= reset_limit:
                        missing = ", ".join(pending_tickers)
                        log(
                            f"Reached max yfinance resets ({reset_limit}) after rate-limit event. "
                            f"Missing {len(pending_tickers)} tickers: {missing}",
                            type="warning",
                        )
                        break

                    completed_count = total_tickers - len(pending_tickers)
                    if len(pending_tickers) < 20:
                        log(
                            f"{completed_count}/{total_tickers} tickers completed for '{client_method}'. "
                            f"{len(pending_tickers)} remaining (<20), retrying immediately without wait."
                        )
                    else:
                        resume_at = datetime.now() + timedelta(seconds=wait)
                        log(
                            f"{completed_count}/{total_tickers} tickers completed for '{client_method}'. "
                            f"Resetting yfinance in {wait} seconds "
                            f"(next retry at {resume_at.strftime('%H:%M:%S')})."
                        )
                        self._sleep(wait)
                    reset_attempts += 1
                    continue
            finally:
                self.yahoo_client = original_client

            if not attempt_df.empty:
                frames.append(attempt_df)
                consecutive_empty_attempts = 0
            else:
                consecutive_empty_attempts += 1

            completed = self._extract_returned_tickers(attempt_df)
            pending_tickers = [
                ticker for ticker in pending_tickers if ticker not in completed
            ]

            if pending_tickers:
                fast_retry = len(pending_tickers) < 20
                if consecutive_empty_attempts >= 2:
                    missing = ", ".join(pending_tickers)
                    log(
                        "Stopping retries after two consecutive empty pull results. "
                        f"Missing {len(pending_tickers)} tickers: {missing}",
                        type="warning",
                    )
                    break
                if reset_attempts >= reset_limit:
                    missing = ", ".join(pending_tickers)
                    log(
                        f"Reached max yfinance resets ({reset_limit}). "
                        f"Missing {len(pending_tickers)} tickers: {missing}",
                        type="warning",
                    )
                    break

                completed_count = total_tickers - len(pending_tickers)
                if fast_retry:
                    log(
                        f"{completed_count}/{total_tickers} tickers completed for '{client_method}'. "
                        f"{len(pending_tickers)} remaining (<20), retrying immediately without wait."
                    )
                    log(f"Retrying '{client_method}' now.")
                else:
                    resume_at = datetime.now() + timedelta(seconds=wait)
                    log(
                        f"{completed_count}/{total_tickers} tickers completed for '{client_method}'. "
                        f"Resetting yfinance in {wait} seconds "
                        f"(next retry at {resume_at.strftime('%H:%M:%S')})."
                    )
                    self._sleep(wait)
                    log(f"Retrying '{client_method}' now.")
                reset_attempts += 1

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _attach_figi_from_mapping(
        dataframe: pd.DataFrame,
        ticker_to_figi: dict[str, str | None] | None,
    ) -> pd.DataFrame:
        """Attach FIGI by TICKER using an external ticker->FIGI mapping."""
        if dataframe.empty or "TICKER" not in dataframe.columns or not ticker_to_figi:
            return dataframe

        out = dataframe.copy()
        out["TICKER"] = out["TICKER"].astype(str).str.strip().str.upper()
        normalized_map = {
            str(ticker).strip().upper(): figi
            for ticker, figi in ticker_to_figi.items()
            if str(ticker).strip()
        }
        mapped_figi = out["TICKER"].map(normalized_map)
        if "FIGI" in out.columns:
            out["FIGI"] = out["FIGI"].where(out["FIGI"].notna(), mapped_figi)
        else:
            out["FIGI"] = mapped_figi
        return out

    @staticmethod
    def _to_signature(value: Any) -> str:
        """Convert values to stable string signatures for row-level comparisons."""
        if pd.isna(value):
            return "<NA>"

        if isinstance(value, str):
            normalized = value.strip()
            lowered = normalized.lower()
            if lowered in {"", "nan", "none", "nat", "null"}:
                return "<NA>"
            if lowered in {"true", "false"}:
                return "1" if lowered == "true" else "0"

            parsed_datetime = pd.to_datetime(normalized, errors="coerce")
            if not pd.isna(parsed_datetime):
                if getattr(parsed_datetime, "tzinfo", None) is not None:
                    parsed_datetime = parsed_datetime.tz_localize(None)
                return parsed_datetime.date().isoformat()

            parsed_numeric = pd.to_numeric(normalized, errors="coerce")
            if not pd.isna(parsed_numeric):
                float_value = float(parsed_numeric)
                if float_value.is_integer():
                    return str(int(float_value))
                return f"{float_value:.15g}"
            return normalized

        if isinstance(value, pd.Timestamp):
            if value.tzinfo is not None:
                value = value.tz_convert(None)
            return value.date().isoformat()
        if isinstance(value, datetime):
            ts = pd.Timestamp(value)
            if ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            return ts.date().isoformat()
        if isinstance(value, dt_date):
            return value.isoformat()
        if isinstance(value, (bool, np.bool_)):
            return "1" if bool(value) else "0"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            float_value = float(value)
            if float_value.is_integer():
                return str(int(float_value))
            return f"{float_value:.15g}"
        return str(value)

    def _filter_new_or_changed_rows(
        self,
        incoming_df: pd.DataFrame,
        existing_df: pd.DataFrame,
        *,
        exclude_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        """Return rows that are not already present when compared on selected columns.

        Parameters
        ----------
        incoming_df
            Candidate rows for writing.
        existing_df
            Existing rows already stored in the destination table.
        exclude_columns
            Column names excluded from change detection (case-insensitive), e.g. ``{"DATE"}``.
        """
        if incoming_df.empty:
            return incoming_df
        if existing_df.empty:
            return incoming_df.drop_duplicates().reset_index(drop=True)

        excluded = {name.upper() for name in (exclude_columns or set())}
        compare_columns = [
            column
            for column in incoming_df.columns
            if column in existing_df.columns and column.upper() not in excluded
        ]
        if not compare_columns:
            return incoming_df.drop_duplicates().reset_index(drop=True)

        existing_keys: set[tuple[str, ...]] = set(
            existing_df[compare_columns]
            .apply(
                lambda row: tuple(self._to_signature(value) for value in row), axis=1
            )
            .tolist()
        )

        keep_indices: list[int] = []
        for idx, row in incoming_df[compare_columns].iterrows():
            key = tuple(self._to_signature(value) for value in row)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            keep_indices.append(idx)

        return incoming_df.loc[keep_indices].reset_index(drop=True)

    def _load_existing_rows_for_figi(
        self,
        *,
        engine,
        table_name: str,
        figis: Iterable[str] | str | None = None,
        figi_column: str = "FIGI",
        log_context: str | None = None,
    ) -> pd.DataFrame:
        """Load existing table rows filtered by FIGI list for delta comparisons."""
        figi_values = self._normalize_filter_values(figis)
        if not figi_values:
            return pd.DataFrame()

        figi_filter = self.sql_client.add_in_filter(
            self.sql_client.quote_ident(figi_column),
            figi_values,
        )
        query = self.sql_client.build_select_with_filters_query(
            table_name=table_name,
            filters_sql=figi_filter,
        )
        context = log_context or table_name
        try:
            out = self.azure_data_source.read_sql_table(
                engine=engine,
                query=query,
                coerce_numeric=False,
            )
            log(f"Loaded existing rows for '{context}' diff check: {len(out)}")
            return out
        except Exception as exc:  # noqa: BLE001
            log(
                f"Could not load existing rows for '{context}' diff check ({exc}). "
                "Proceeding with full write payload.",
                type="warning",
            )
            return pd.DataFrame()
