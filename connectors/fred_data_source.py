"""FRED/ALFRED data-loading client.

This module mirrors the shape of :mod:`connectors.yahoo_data_source` while
supporting revision-aware macroeconomic data pulls from the FRED API.
"""

import time
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import requests

import utils.dataframe_utils as dataframe_utils
from config.configs import Configs
from utils.logging import log
from utils.threading import TaskExecutionError, ThreadWorkerPool

SeriesFetcherFn = Callable[[str], pd.DataFrame | None]


class FredDataClient:
    """Threaded FRED client for revision-aware macro series extraction.

    Parameters
    ----------
    series_codes : sequence[str]
        FRED series identifiers to query.
    max_workers : int, default 4
        Maximum number of worker threads used for parallel series fetches.
    retries : int, default 3
        Number of retry attempts for a failed request.
    timeout : int, default 30
        Request timeout in seconds.
    session : requests.Session | None, default None
        Optional injected HTTP session for testing or custom transport policy.
    """

    BASE_URL = "https://api.stlouisfed.org/fred"
    EARLIEST_REALTIME_START = "1776-07-04"
    LATEST_REALTIME_END = "9999-12-31"
    RATE_LIMIT_SLEEP_SECONDS = 60

    def __init__(
        self,
        series_codes: Sequence[str],
        *,
        max_workers: int = 4,
        retries: int = 3,
        timeout: int = 30,
        session: requests.Session | None = None,
        configs_path: str | None = None,
    ) -> None:
        if not isinstance(series_codes, Sequence) or isinstance(
            series_codes, (str, bytes)
        ):
            raise ValueError("series_codes must be a sequence of series code strings")

        cleaned_codes = [
            str(series_code).strip().upper()
            for series_code in series_codes
            if str(series_code).strip()
        ]
        if not cleaned_codes:
            raise ValueError("series_codes cannot be empty")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if retries < 1:
            raise ValueError("retries must be >= 1")
        if timeout < 1:
            raise ValueError("timeout must be >= 1")

        self.series_codes = list(dict.fromkeys(cleaned_codes))
        self.api_key = self._resolve_api_key(configs_path=configs_path)
        if not self.api_key:
            raise ValueError("A FRED API key is required.")

        self.max_workers = max_workers
        self.retries = retries
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self.pool = ThreadWorkerPool(max_workers=max_workers)

    @staticmethod
    def _default_configs_path() -> str:
        """Return the default repository config path for FRED credentials."""
        return str(
            (
                Path(__file__).resolve().parent.parent / "config" / "configs.json"
            ).resolve()
        )

    @classmethod
    def _load_fred_config(cls, configs_path: str | None = None) -> dict[str, Any]:
        """Load the FRED config section from the repository config file."""
        config_path = configs_path or cls._default_configs_path()
        configs = Configs(path=config_path).load().as_dict()

        try:
            fred_cfg = configs["fred"]
        except KeyError as exc:
            raise KeyError('Config missing required top-level key "fred".') from exc

        if not isinstance(fred_cfg, dict):
            raise ValueError("Config key 'fred' must be an object.")
        return fred_cfg

    @classmethod
    def _resolve_api_key(
        cls,
        *,
        configs_path: str | None,
    ) -> str | None:
        """Resolve the FRED API key from the repository config."""
        fred_cfg = cls._load_fred_config(configs_path=configs_path)
        resolved = fred_cfg.get("api_key")
        if not resolved or not str(resolved).strip():
            raise ValueError("Missing FRED API key in config: fred.api_key")
        return str(resolved).strip()

    def _retry_fetch(
        self, func: Callable[[], dict[str, Any]], label: str
    ) -> dict[str, Any]:
        """Execute an HTTP fetch function with retry and consistent backoff."""
        for attempt in range(1, self.retries + 1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.retries or not self._is_retriable_error(exc):
                    raise
                delay_seconds = self._retry_delay_seconds(exc, attempt)
                if delay_seconds >= self.RATE_LIMIT_SLEEP_SECONDS:
                    log(
                        "FRED rate limit hit; retrying after "
                        f"{int(delay_seconds)} seconds "
                        f"(attempt {attempt}/{self.retries}) for {label}.",
                        type="warning",
                    )
                time.sleep(delay_seconds)
        raise RuntimeError(f"Unexpected retry exhaustion while loading {label}")

    @staticmethod
    def _is_retriable_error(exc: Exception) -> bool:
        """Return True when an exception indicates a temporary HTTP/API failure."""
        message = str(exc).lower()
        retriable_markers = (
            "too many requests",
            "rate limit",
            "temporarily unavailable",
            "timed out",
            "connection reset",
            "502",
            "503",
            "504",
            "429",
        )
        return any(marker in message for marker in retriable_markers)

    @classmethod
    def _retry_delay_seconds(cls, exc: Exception, attempt: int) -> float:
        """Return retry delay seconds for a temporary API failure."""
        message = str(exc).lower()
        if (
            "429" in message
            or "rate limit" in message
            or "too many requests" in message
        ):
            return float(cls.RATE_LIMIT_SLEEP_SECONDS)
        return min(2.0, 0.7 * attempt)

    def _request_json(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        limit: int | None = None,
        max_limit: int = 100000,
    ) -> dict[str, Any]:
        """Execute a GET request to the FRED API and return parsed JSON."""
        payload = dict(params or {})
        payload["api_key"] = self.api_key
        payload["file_type"] = "json"
        if limit is not None:
            payload["limit"] = min(int(limit), max_limit)

        def inner() -> dict[str, Any]:
            response = self.session.get(
                f"{self.BASE_URL}/{endpoint}",
                params=payload,
                timeout=self.timeout,
            )
            if response.status_code == 429:
                raise RuntimeError("FRED rate limit hit (429 Too Many Requests)")
            try:
                body = response.json()
            except ValueError:
                body = None
            if response.status_code >= 400:
                if isinstance(body, dict) and body.get("error_code"):
                    raise RuntimeError(
                        f"FRED API error {body.get('error_code')}: "
                        f"{body.get('error_message')}"
                    )
                response.raise_for_status()
            if body is None:
                raise ValueError(
                    f"Expected JSON response body from FRED endpoint '{endpoint}'."
                )
            if isinstance(body, dict) and body.get("error_code"):
                raise RuntimeError(
                    f"FRED API error {body.get('error_code')}: {body.get('error_message')}"
                )
            return body

        return self._retry_fetch(inner, endpoint)

    def _request_all_pages(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        items_key: str,
        page_limit: int,
    ) -> list[Any]:
        """Fetch all paginated records for an endpoint using FRED count/offset."""
        collected: list[Any] = []
        offset = 0

        while True:
            body = self._request_json(
                endpoint,
                params={**(params or {}), "offset": offset},
                limit=page_limit,
                max_limit=page_limit,
            )
            items = body.get(items_key, [])
            if not isinstance(items, list):
                raise ValueError(
                    f"Expected '{items_key}' list in FRED response for endpoint '{endpoint}'."
                )
            collected.extend(items)

            count = int(body.get("count", len(collected)))
            limit = int(body.get("limit", page_limit))
            offset = int(body.get("offset", offset)) + limit
            if len(collected) >= count or not items:
                break

        return collected

    @staticmethod
    def _concat_results(results: Sequence[pd.DataFrame | None]) -> pd.DataFrame:
        """Concatenate non-empty DataFrames and return empty DataFrame when none exist."""
        frames = [frame for frame in results if frame is not None and not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame columns and enforce uniqueness after normalization."""
        out = dataframe_utils.normalize_columns(df)
        if out.columns.is_unique:
            return out

        deduped: list[str] = []
        seen: dict[str, int] = {}
        for col in out.columns:
            count = seen.get(col, 0)
            seen[col] = count + 1
            deduped.append(col if count == 0 else f"{col}_{count + 1}")
        out.columns = deduped
        return out

    @staticmethod
    def _to_date_series(series: pd.Series) -> pd.Series:
        """Convert a date-like series into python date objects."""
        return dataframe_utils.coerce_datetime_series(series, errors="coerce").dt.date

    @staticmethod
    def _normalize_value_series(series: pd.Series) -> pd.Series:
        """Convert FRED value strings to floats."""
        cleaned = series.astype("string").str.strip()
        missing_mask = cleaned.isin({"", ".", "nan", "None", "none", "<NA>"})
        numeric = pd.to_numeric(cleaned.mask(missing_mask), errors="coerce")
        return numeric.astype(float)

    @staticmethod
    def _normalize_observation_rows(
        rows: list[dict[str, Any]],
        *,
        series_code: str,
    ) -> pd.DataFrame:
        """Normalize raw FRED observation rows into a research-grade table."""
        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows)
        out = FredDataClient._normalize(out)
        value_series = FredDataClient._normalize_value_series(
            out.get("VALUE", pd.Series(dtype="object"))
        )
        out["VALUE"] = value_series
        out["OBSERVATION_DATE"] = FredDataClient._to_date_series(out["DATE"])
        out["REALTIME_START"] = FredDataClient._to_date_series(out["REALTIME_START"])
        out["REALTIME_END"] = FredDataClient._to_date_series(out["REALTIME_END"])
        out["TICKER"] = str(series_code).strip().upper()
        out = out.loc[out["VALUE"].notna()].copy()
        out = out.drop(columns=["DATE"], errors="ignore")
        first_columns = [
            "TICKER",
            "OBSERVATION_DATE",
            "REALTIME_START",
            "REALTIME_END",
            "VALUE",
        ]
        return out[first_columns + [c for c in out.columns if c not in first_columns]]

    @staticmethod
    def _normalize_metadata_rows(
        rows: list[dict[str, Any]],
        *,
        series_code: str,
    ) -> pd.DataFrame:
        """Normalize raw FRED series metadata rows."""
        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows)
        out = FredDataClient._normalize(out)
        if "ID" in out.columns:
            out["TICKER"] = out["ID"].astype(str).str.strip().str.upper()
            out = out.drop(columns=["ID"])
        else:
            out["TICKER"] = str(series_code).strip().upper()

        for column in ("OBSERVATION_START",):
            if column in out.columns:
                out[column] = FredDataClient._to_date_series(out[column])
        out = out.drop(
            columns=[
                "OBSERVATION_END",
                "REALTIME_START",
                "REALTIME_END",
                "LAST_UPDATED",
            ],
            errors="ignore",
        )
        first_columns = ["TICKER"]
        return out[first_columns + [c for c in out.columns if c not in first_columns]]

    @staticmethod
    def _normalize_vintage_date_rows(
        rows: list[Any],
        *,
        series_code: str,
    ) -> pd.DataFrame:
        """Normalize FRED vintage date payloads to a tabular form."""
        if not rows:
            return pd.DataFrame(columns=["TICKER", "VINTAGE_DATE"])

        out = pd.DataFrame({"VINTAGE_DATE": rows})
        out["TICKER"] = str(series_code).strip().upper()
        out["VINTAGE_DATE"] = FredDataClient._to_date_series(out["VINTAGE_DATE"])
        return out[["TICKER", "VINTAGE_DATE"]]

    def _run_parallel(
        self,
        fetcher: SeriesFetcherFn,
        log_message: str | None,
        series_codes: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Run a series fetcher concurrently for all or selected series."""
        codes = self.series_codes if series_codes is None else list(series_codes)
        if log_message:
            log(log_message)
        tasks = [(lambda code=series_code: fetcher(code)) for series_code in codes]
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
            retriable_error = RuntimeError(str(exc.original_exception))
            setattr(retriable_error, "partial_df", partial_df)
            raise retriable_error from exc.original_exception

        for result in results:
            if isinstance(result, Exception):
                raise result
        return self._concat_results(results)

    def get_series_metadata(self, series_code: str) -> pd.DataFrame:
        """Return normalized metadata for a single FRED series."""
        body = self._request_json("series", params={"series_id": series_code})
        rows = body.get("seriess", [])
        return self._normalize_metadata_rows(
            rows,
            series_code=series_code,
        )

    def get_many_series_metadata(
        self,
        series_codes: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return normalized metadata for many FRED series."""
        return self._run_parallel(
            self.get_series_metadata,
            "Loading FRED Series Metadata",
            series_codes=series_codes,
        )

    def get_series_observations(
        self,
        series_code: str,
        *,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        observation_start: str | None = None,
        observation_end: str | None = None,
        units: str | None = None,
        frequency: str | None = None,
        aggregation_method: str | None = None,
        output_type: int = 1,
        vintage_dates: str | None = None,
    ) -> pd.DataFrame:
        """Return normalized observations for a single series.

        Notes
        -----
        ``output_type=1`` with a broad real-time window preserves each returned
        row's ``REALTIME_START`` and ``REALTIME_END`` validity interval, which is
        the foundation for the loader's vintage-aware storage model.
        """
        params = {
            "series_id": str(series_code).strip().upper(),
            "output_type": output_type,
        }
        optional_values = {
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
            "observation_start": observation_start,
            "observation_end": observation_end,
            "units": units,
            "frequency": frequency,
            "aggregation_method": aggregation_method,
            "vintage_dates": vintage_dates,
        }
        params.update(
            {
                key: value
                for key, value in optional_values.items()
                if value is not None and str(value).strip() != ""
            }
        )
        rows = self._request_all_pages(
            "series/observations",
            params=params,
            items_key="observations",
            page_limit=100000,
        )
        return self._normalize_observation_rows(
            rows,
            series_code=series_code,
        )

    def get_series_vintage_observations(
        self,
        series_code: str,
        *,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        observation_start: str | None = None,
        observation_end: str | None = None,
        units: str | None = None,
        frequency: str | None = None,
        aggregation_method: str | None = None,
    ) -> pd.DataFrame:
        """Return revision-aware observations across the full available real-time span."""
        return self.get_series_observations(
            series_code,
            realtime_start=realtime_start or self.EARLIEST_REALTIME_START,
            realtime_end=realtime_end or self.LATEST_REALTIME_END,
            observation_start=observation_start,
            observation_end=observation_end,
            units=units,
            frequency=frequency,
            aggregation_method=aggregation_method,
            output_type=1,
        )

    def get_series_vintages(
        self,
        series_code: str,
        *,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.DataFrame:
        """Return vintage dates for a single FRED series."""
        rows = self._request_all_pages(
            "series/vintagedates",
            params={
                "series_id": str(series_code).strip().upper(),
                **(
                    {"realtime_start": realtime_start}
                    if realtime_start is not None
                    else {}
                ),
                **({"realtime_end": realtime_end} if realtime_end is not None else {}),
            },
            items_key="vintage_dates",
            page_limit=10000,
        )
        return self._normalize_vintage_date_rows(
            rows,
            series_code=series_code,
        )

    def get_many_series(
        self,
        series_codes: Sequence[str] | None = None,
        *,
        series_options: dict[str, dict[str, Any]] | None = None,
        include_revisions: bool = True,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pd.DataFrame:
        """Return observations for many series using optional per-series overrides."""

        def fetch_one(series_code: str) -> pd.DataFrame:
            options = dict((series_options or {}).get(series_code, {}))
            options.setdefault("realtime_start", realtime_start)
            options.setdefault("realtime_end", realtime_end)
            options.setdefault("observation_start", observation_start)
            options.setdefault("observation_end", observation_end)
            if include_revisions:
                return self.get_series_vintage_observations(series_code, **options)
            return self.get_series_observations(series_code, **options)

        return self._run_parallel(
            fetch_one,
            None,
            series_codes=series_codes,
        )


__all__ = ["FredDataClient"]
