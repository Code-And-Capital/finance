"""Config-driven FRED macroeconomic data pipeline."""

from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import inspect
from sqlalchemy import types as satypes

import utils.dataframe_utils as dataframe_utils
from config.configs import Configs
from connectors.azure_data_source import default_azure_data_source
from connectors.fred_data_source import FredDataClient
from sql.script_factory import SQLClient
from utils.logging import log

_DEFAULT_FRED_SERIES_GROUPS: dict[str, list[str]] = {
    "rates": [
        "FEDFUNDS",
        "DFF",
        "DTB3",
        "DGS2",
        "DGS5",
        "DGS10",
        "DGS30",
        "T10Y2Y",
        "T10Y3M",
    ],
    "inflation": [
        "CPIAUCSL",
        "CPILFESL",
        "PCEPI",
        "PCEPILFE",
        "T5YIE",
        "T10YIE",
    ],
    "labor": [
        "UNRATE",
        "U6RATE",
        "PAYEMS",
        "ICSA",
        "CIVPART",
        "AWHAETP",
    ],
    "growth": [
        "GDPC1",
        "INDPRO",
        "TCU",
        "BUSINV",
        "ISRATIO",
    ],
    "consumer": [
        "UMCSENT",
        "CSUSHPISA",
        "RSAFS",
    ],
    "housing": [
        "HOUST",
        "PERMIT",
        "HSN1F",
        "MORTGAGE30US",
    ],
    "credit": [
        "BAA",
        "AAA",
        "BAA10Y",
        "BAMLH0A0HYM2",
        "STLFSI4",
    ],
    "liquidity": [
        "M1SL",
        "M2SL",
        "WALCL",
    ],
    "commodities": [
        "DCOILWTICO",
    ],
    "trade": [
        "DTWEXBGS",
    ],
    "recession": [
        "USREC",
    ],
}


def _default_series_configs() -> list[dict[str, Any]]:
    """Return the built-in FRED series catalog used when config omits one."""
    rows: list[dict[str, Any]] = []
    for category, series_codes in _DEFAULT_FRED_SERIES_GROUPS.items():
        for series_code in series_codes:
            rows.append(
                {
                    "series_code": series_code,
                    "category": category,
                    "frequency": None,
                    "units": None,
                    "active": True,
                }
            )
    return rows


class FredData:
    """Config-driven pipeline for FRED metadata and economic data history.

    Daily series are stored as stable latest-only history. Weekly and lower
    frequency series are stored revision-aware using FRED real-time intervals.
    """

    DEFAULT_LOOKBACK_DAYS = 365
    DEFAULT_BATCH_YEARS = 1

    def __init__(
        self,
        *,
        series_configs: list[dict[str, Any]] | None = None,
        series_codes: Iterable[str] | str | None = None,
        fred_client=None,
        max_workers: int = 4,
    ) -> None:
        self.series_codes = self._normalize_series_codes(series_codes)
        self.series_configs = series_configs
        self.fred_client = fred_client
        self.max_workers = max_workers
        self._cached_metadata_df: pd.DataFrame | None = None
        self.clients_used: list[object] = []
        self.azure_data_source = default_azure_data_source
        self.sql_client = SQLClient()
        self.table_name_metadata = "fred_series"
        self.table_name_revised_data = "fred_economic_data"
        self._today = date.today

    @staticmethod
    def _normalize_series_codes(
        series_codes: Iterable[str] | str | None,
    ) -> list[str]:
        """Normalize requested series codes to distinct uppercase values."""
        if series_codes is None:
            return []
        if isinstance(series_codes, str):
            raw = [series_codes]
        else:
            raw = list(series_codes)
        return list(
            dict.fromkeys(
                str(series_code).strip().upper()
                for series_code in raw
                if str(series_code).strip()
            )
        )

    def _track_client(self, client) -> None:
        """Track a client instance once for lifecycle introspection."""
        if not any(existing is client for existing in self.clients_used):
            self.clients_used.append(client)

    @staticmethod
    def _default_configs_path() -> str:
        """Return the default repository config path."""
        return FredDataClient._default_configs_path()

    def _load_fred_settings(self, configs_path: str | None = None) -> dict[str, Any]:
        """Load the ``fred`` section from the repo config file."""
        path = configs_path or self._default_configs_path()
        configs = Configs(path=path).load().as_dict()
        fred_cfg = configs.get("fred", {})
        if not isinstance(fred_cfg, dict):
            raise ValueError("Config key 'fred' must be an object.")
        return fred_cfg

    def _normalize_series_config(self, item: dict[str, Any]) -> dict[str, Any]:
        """Normalize one configured series entry to the pipeline schema."""
        series_code = (
            str(item.get("series_code") or item.get("code") or item.get("id") or "")
            .strip()
            .upper()
        )
        if not series_code:
            raise ValueError(
                "Each FRED series config requires a non-empty series_code."
            )

        active_raw = item.get("active", True)
        active = (
            bool(active_raw)
            if not isinstance(active_raw, str)
            else (active_raw.strip().lower() not in {"0", "false", "no", "n"})
        )

        return {
            "TICKER": series_code,
            "CATEGORY": item.get("category"),
            "FREQUENCY_PREFERENCE": item.get("frequency"),
            "UNITS_PREFERENCE": item.get("units"),
            "IS_ACTIVE": active,
        }

    def _get_series_config_dataframe(
        self,
        *,
        configs_path: str | None = None,
    ) -> pd.DataFrame:
        """Return the active configured FRED series catalog."""
        configured = self.series_configs
        if configured is None:
            fred_cfg = self._load_fred_settings(configs_path=configs_path)
            configured = fred_cfg.get("series") or _default_series_configs()

        if not isinstance(configured, list):
            raise ValueError("fred.series must be a list when provided in config.")

        rows = [self._normalize_series_config(item) for item in configured]
        out = pd.DataFrame(rows).drop_duplicates(subset=["TICKER"], keep="first")
        if self.series_codes:
            out = out[out["TICKER"].isin(set(self.series_codes))]
        out = out[out["IS_ACTIVE"]].copy()
        if out.empty:
            raise ValueError("No active FRED series are configured for loading.")
        return out.reset_index(drop=True)

    def _create_client(
        self,
        *,
        configs_path: str | None = None,
        series_codes: Iterable[str] | None = None,
    ) -> FredDataClient:
        """Create a FredDataClient for configured series."""
        fred_cfg = self._load_fred_settings(configs_path=configs_path)
        codes = self._normalize_series_codes(
            series_codes
        ) or self._normalize_series_codes(
            self._get_series_config_dataframe(configs_path=configs_path)["TICKER"]
        )
        client = FredDataClient(
            codes,
            max_workers=int(fred_cfg.get("max_workers", self.max_workers)),
            retries=int(fred_cfg.get("retries", 3)),
            timeout=int(fred_cfg.get("timeout_seconds", 30)),
            configs_path=configs_path,
        )
        self._track_client(client)
        return client

    def _resolve_client(self, *, configs_path: str | None = None) -> FredDataClient:
        """Resolve an injected FRED client or create a default one."""
        if self.fred_client is not None:
            self._track_client(self.fred_client)
            return self.fred_client
        self.fred_client = self._create_client(configs_path=configs_path)
        return self.fred_client

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
            return normalized

        if isinstance(value, pd.Timestamp):
            if value.tzinfo is not None:
                value = value.tz_convert(None)
            return value.isoformat()
        if isinstance(value, (date,)):
            return value.isoformat()
        return str(value)

    def _series_options_from_config(
        self, config_df: pd.DataFrame
    ) -> dict[str, dict[str, Any]]:
        """Translate config rows into per-series FRED request overrides."""
        options: dict[str, dict[str, Any]] = {}
        for row in config_df.itertuples(index=False):
            series_options = {
                "units": row.UNITS_PREFERENCE,
                "frequency": row.FREQUENCY_PREFERENCE,
            }
            options[row.TICKER] = {
                key: value
                for key, value in series_options.items()
                if value is not None and str(value).strip() != ""
            }
        return options

    def _merge_series_metadata(
        self,
        metadata_df: pd.DataFrame,
        config_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge API metadata with local config metadata."""
        config_cols = [
            "TICKER",
            "CATEGORY",
            "IS_ACTIVE",
        ]
        merged = config_df[config_cols].merge(
            metadata_df,
            how="left",
            on="TICKER",
        )
        merged["SOURCE"] = "FRED"
        ordered_columns = [
            "TICKER",
            "TITLE",
            "CATEGORY",
            "FREQUENCY",
            "UNITS",
            "SEASONAL_ADJUSTMENT",
            "NOTES",
            "SOURCE",
            "IS_ACTIVE",
            "POPULARITY",
            "OBSERVATION_START",
            "FREQUENCY_SHORT",
            "UNITS_SHORT",
            "SEASONAL_ADJUSTMENT_SHORT",
        ]
        for column in ordered_columns:
            if column not in merged.columns:
                merged[column] = None
        return merged[ordered_columns]

    def _update_metadata_cache(self, metadata_df: pd.DataFrame) -> None:
        """Merge fresh metadata rows into the instance cache by ticker."""
        if metadata_df.empty:
            return
        incoming = metadata_df.copy()
        if self._cached_metadata_df is None or self._cached_metadata_df.empty:
            self._cached_metadata_df = incoming.reset_index(drop=True)
            return
        combined = pd.concat([self._cached_metadata_df, incoming], ignore_index=True)
        combined["TICKER"] = combined["TICKER"].astype(str).str.strip().str.upper()
        self._cached_metadata_df = combined.drop_duplicates(
            subset=["TICKER"], keep="last"
        ).reset_index(drop=True)

    def _get_cached_metadata(
        self,
        *,
        tickers: list[str],
    ) -> pd.DataFrame | None:
        """Return cached metadata subset when all requested tickers are present."""
        if self._cached_metadata_df is None or self._cached_metadata_df.empty:
            return None
        cached = self._cached_metadata_df.copy()
        cached["TICKER"] = cached["TICKER"].astype(str).str.strip().str.upper()
        requested = [str(ticker).strip().upper() for ticker in tickers]
        subset = cached[cached["TICKER"].isin(set(requested))].copy()
        if subset["TICKER"].nunique() != len(set(requested)):
            return None
        return subset.reset_index(drop=True)

    def _resolve_observation_start(
        self,
        *,
        full_history: bool,
        configs_path: str | None,
        start_date: str | None,
        lookback_days: int | None,
    ) -> str | None:
        """Resolve the observation-date lower bound for pipeline pulls."""
        if full_history:
            return start_date
        if start_date is not None:
            return pd.Timestamp(start_date).date().isoformat()

        fred_cfg = self._load_fred_settings(configs_path=configs_path)
        resolved_lookback = int(
            lookback_days
            if lookback_days is not None
            else fred_cfg.get("revisions_lookback_days", self.DEFAULT_LOOKBACK_DAYS)
        )
        start_day = self._today() - timedelta(days=resolved_lookback)
        return start_day.isoformat()

    def _resolve_observation_end(self, end_date: str | None) -> str:
        """Resolve the inclusive upper-bound observation date for batched pulls."""
        if end_date is not None:
            return pd.Timestamp(end_date).date().isoformat()
        return self._today().isoformat()

    def _resolve_full_history_start(
        self,
        *,
        metadata_df: pd.DataFrame,
        start_date: str | None,
    ) -> str:
        """Resolve the first available history date from metadata."""
        if start_date is not None:
            return pd.Timestamp(start_date).date().isoformat()

        try:
            if "OBSERVATION_START" in metadata_df.columns and not metadata_df.empty:
                starts = dataframe_utils.coerce_datetime_series(
                    metadata_df["OBSERVATION_START"],
                    errors="coerce",
                ).dropna()
                if not starts.empty:
                    return starts.min().date().isoformat()
        except Exception as exc:  # noqa: BLE001
            log(
                "Could not resolve full-history FRED start date from metadata "
                f"({exc}). Falling back to 1900-01-01.",
                type="warning",
            )

        return "1900-01-01"

    @staticmethod
    def _metadata_observation_start_lookup(
        metadata_df: pd.DataFrame,
    ) -> dict[str, date]:
        """Return per-ticker observation-start dates from metadata."""
        if metadata_df.empty or "TICKER" not in metadata_df.columns:
            return {}
        if "OBSERVATION_START" not in metadata_df.columns:
            return {}

        starts = dataframe_utils.coerce_datetime_series(
            metadata_df["OBSERVATION_START"],
            errors="coerce",
        ).dt.date
        out: dict[str, date] = {}
        for ticker, observation_start in zip(metadata_df["TICKER"], starts):
            if pd.isna(observation_start):
                continue
            out[str(ticker).strip().upper()] = observation_start
        return out

    @staticmethod
    def _frequency_lookup(metadata_df: pd.DataFrame) -> dict[str, str]:
        """Return normalized frequency labels by ticker."""
        if metadata_df.empty or "TICKER" not in metadata_df.columns:
            return {}

        out: dict[str, str] = {}
        for _, row in metadata_df.iterrows():
            frequency = row.get("FREQUENCY_SHORT")
            if pd.isna(frequency) or str(frequency).strip() == "":
                frequency = row.get("FREQUENCY")
            if pd.isna(frequency) or str(frequency).strip() == "":
                continue
            out[str(row["TICKER"]).strip().upper()] = str(frequency).strip().lower()
        return out

    @classmethod
    def _split_tickers_by_frequency(
        cls,
        *,
        tickers: list[str],
        metadata_df: pd.DataFrame,
    ) -> tuple[list[str], list[str]]:
        """Split tickers into daily latest-only and revised-data groups."""
        frequency_lookup = cls._frequency_lookup(metadata_df)
        daily_tickers: list[str] = []
        revised_tickers: list[str] = []
        for ticker in tickers:
            frequency = frequency_lookup.get(ticker, "")
            if frequency in {"d", "daily"}:
                daily_tickers.append(ticker)
            else:
                revised_tickers.append(ticker)
        return daily_tickers, revised_tickers

    @staticmethod
    def _build_observation_batches(
        *,
        start_date: str,
        end_date: str,
        batch_years: int,
    ) -> list[tuple[str, str]]:
        """Split an inclusive date range into yearly batches."""
        if batch_years < 1:
            raise ValueError("batch_years must be >= 1")

        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()
        if start_ts > end_ts:
            raise ValueError("start_date must be <= end_date")

        windows: list[tuple[str, str]] = []
        current_start = start_ts
        while current_start <= end_ts:
            if current_start.year + batch_years > 9999:
                current_end = end_ts
            else:
                current_end = min(
                    current_start
                    + pd.DateOffset(years=batch_years)
                    - pd.Timedelta(days=1),
                    end_ts,
                )
            windows.append(
                (current_start.date().isoformat(), current_end.date().isoformat())
            )
            current_start = current_end + pd.Timedelta(days=1)
        return windows

    @staticmethod
    def _normalize_daily_latest_data(
        revised_data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Stabilize latest-only rows for storage.

        Latest-only FRED pulls reflect the current real-time period, which would
        otherwise make every row look new on each run. We therefore anchor these
        rows to their observation date so group-level write comparisons remain
        stable.
        """
        if revised_data_df.empty:
            return revised_data_df
        out = revised_data_df.copy()
        out["REALTIME_START"] = out["OBSERVATION_DATE"]
        out["REALTIME_END"] = out["OBSERVATION_DATE"]
        return out

    @staticmethod
    def _normalize_revised_data(
        revised_data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize revision-aware observations for research and storage."""
        if revised_data_df.empty:
            return pd.DataFrame(
                columns=[
                    "TICKER",
                    "OBSERVATION_DATE",
                    "VALUE",
                    "REALTIME_START",
                    "REALTIME_END",
                ]
            )

        out = revised_data_df.copy()
        out["VALUE"] = pd.to_numeric(out["VALUE"], errors="coerce").astype(float)
        for column in ("OBSERVATION_DATE", "REALTIME_START", "REALTIME_END"):
            out[column] = dataframe_utils.coerce_datetime_series(
                out[column], errors="coerce"
            ).dt.date
        out["TICKER"] = out["TICKER"].astype(str).str.strip().str.upper()
        out = out.dropna(subset=["TICKER", "OBSERVATION_DATE"]).reset_index(drop=True)
        hash_columns = [
            "TICKER",
            "OBSERVATION_DATE",
            "VALUE",
            "REALTIME_START",
            "REALTIME_END",
        ]
        extras = [column for column in out.columns if column not in hash_columns]
        return out[hash_columns + extras]

    def _load_existing_rows(
        self,
        *,
        engine,
        table_name: str,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load an existing ticker-scoped table slice for diff checks."""
        filters_sql = ""
        if tickers:
            filters_sql = self.sql_client.add_in_filter(
                self.sql_client.quote_ident("TICKER"),
                tickers,
            )
        query = self.sql_client.build_select_with_filters_query(
            table_name=table_name,
            filters_sql=filters_sql,
        )
        try:
            return self.azure_data_source.read_sql_table(
                engine=engine,
                query=query,
                coerce_numeric=False,
            )
        except Exception as exc:  # noqa: BLE001
            log(
                f"Could not load existing rows for '{table_name}' diff check ({exc}). "
                "Proceeding with write payload as-is.",
                type="warning",
            )
            return pd.DataFrame()

    @staticmethod
    def _filter_existing_to_incoming_groups(
        *,
        existing_df: pd.DataFrame,
        incoming_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Restrict existing rows to the observation groups present in incoming data."""
        if existing_df.empty or incoming_df.empty:
            return existing_df

        incoming_keys = set(
            zip(
                incoming_df["TICKER"].astype(str).str.strip().str.upper(),
                incoming_df["OBSERVATION_DATE"].astype(str),
            )
        )
        existing_keys = list(
            zip(
                existing_df["TICKER"].astype(str).str.strip().str.upper(),
                existing_df["OBSERVATION_DATE"].astype(str),
            )
        )
        mask = [key in incoming_keys for key in existing_keys]
        return existing_df.loc[mask].reset_index(drop=True)

    def _load_run_metadata(
        self,
        *,
        config_df: pd.DataFrame,
        client: FredDataClient,
    ) -> pd.DataFrame:
        """Return cached metadata when possible, otherwise fetch it once."""
        tickers = config_df["TICKER"].tolist()
        metadata_df = self._get_cached_metadata(tickers=tickers)
        if metadata_df is None:
            metadata_df = client.get_many_series_metadata(tickers)
            metadata_df = self._merge_series_metadata(metadata_df, config_df)
            self._update_metadata_cache(metadata_df)
        return metadata_df

    def _load_ticker_batches(
        self,
        *,
        client: FredDataClient,
        ticker: str,
        series_options: dict[str, Any],
        observation_start: str,
        observation_end: str,
        ticker_number: int,
        ticker_count: int,
        batch_years: int,
        include_revisions: bool,
        mode_label: str,
    ) -> pd.DataFrame:
        """Load one ticker across observation-date batches."""
        windows = self._build_observation_batches(
            start_date=observation_start,
            end_date=observation_end,
            batch_years=batch_years,
        )
        frames: list[pd.DataFrame] = []
        for batch_number, (batch_start, batch_end) in enumerate(windows, start=1):
            if len(windows) > 1:
                log(
                    f"FRED {mode_label} batch: "
                    f"ticker={ticker} "
                    f"series={ticker_number}/{ticker_count} "
                    f"batch={batch_number}/{len(windows)} "
                    f"observation_start={batch_start} "
                    f"observation_end={batch_end}"
                )
            batch_df = client.get_many_series(
                [ticker],
                series_options={ticker: dict(series_options)},
                include_revisions=include_revisions,
                observation_start=batch_start,
                observation_end=batch_end,
            )
            if not batch_df.empty:
                frames.append(batch_df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _filter_new_rows(
        incoming_df: pd.DataFrame,
        existing_df: pd.DataFrame,
        *,
        compare_columns: list[str],
    ) -> tuple[pd.DataFrame, int]:
        """Return rows not already present in the destination slice."""
        if incoming_df.empty:
            return incoming_df, 0

        deduped_incoming = incoming_df.drop_duplicates(
            subset=compare_columns
        ).reset_index(drop=True)
        if existing_df.empty:
            skipped = len(incoming_df) - len(deduped_incoming)
            return deduped_incoming, skipped

        comparable_existing = existing_df.copy()
        existing_keys = set(
            comparable_existing[compare_columns]
            .apply(
                lambda row: tuple(FredData._to_signature(value) for value in row),
                axis=1,
            )
            .tolist()
        )

        keep_indices: list[int] = []
        for idx, row in deduped_incoming[compare_columns].iterrows():
            key = tuple(FredData._to_signature(value) for value in row)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            keep_indices.append(idx)

        filtered = deduped_incoming.loc[keep_indices].reset_index(drop=True)
        skipped = len(incoming_df) - len(filtered)
        return filtered, skipped

    def _table_exists(self, *, engine, table_name: str) -> bool:
        """Return True when a table already exists in Azure SQL."""
        try:
            return inspect(engine).has_table(table_name, schema="dbo")
        except Exception:  # noqa: BLE001
            return False

    def _delete_rows(
        self,
        *,
        engine,
        table_name: str,
        where_clause: str,
    ) -> None:
        """Delete a targeted row slice from Azure SQL."""
        query = self.sql_client.build_delete_query(
            table_name=table_name,
            where_clause=where_clause,
            schema="dbo",
        )
        self.azure_data_source.delete_sql_rows(query=query, engine=engine)

    def _build_metadata_where_clause(self, tickers: list[str]) -> str:
        """Build DELETE predicate for metadata rows."""
        escaped = [ticker.replace("'", "''") for ticker in tickers]
        joined = "', '".join(escaped)
        return f"TICKER IN ('{joined}')"

    @staticmethod
    def _group_realtime_start_max(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return max REALTIME_START by `(TICKER, OBSERVATION_DATE)` group."""
        if dataframe.empty:
            return pd.DataFrame(
                columns=[
                    "TICKER_KEY",
                    "OBSERVATION_DATE_KEY",
                    "MAX_REALTIME_START",
                ]
            )

        keyed = dataframe.copy()
        keyed["TICKER_KEY"] = keyed["TICKER"].astype(str).str.strip().str.upper()
        keyed["OBSERVATION_DATE_KEY"] = keyed["OBSERVATION_DATE"].apply(
            lambda value: value.isoformat() if pd.notna(value) else None
        )
        grouped = (
            keyed.groupby(["TICKER_KEY", "OBSERVATION_DATE_KEY"], dropna=False)[
                "REALTIME_START"
            ]
            .max()
            .reset_index()
            .rename(columns={"REALTIME_START": "MAX_REALTIME_START"})
        )
        return grouped

    @classmethod
    def _compute_revised_data_group_refreshes(
        cls,
        incoming_df: pd.DataFrame,
        existing_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, set[tuple[str, str]], set[tuple[str, str]]]:
        """Return rows to write plus groups to replace and insert.

        Logic:
        - if a group is new in incoming data, insert it
        - if incoming max(REALTIME_START) > existing max(REALTIME_START), replace the group
        - otherwise leave the group untouched
        """
        compare_columns = [
            "TICKER",
            "OBSERVATION_DATE",
            "VALUE",
            "REALTIME_START",
            "REALTIME_END",
        ]
        incoming = incoming_df.drop_duplicates(subset=compare_columns).reset_index(
            drop=True
        )
        existing = (
            existing_df.drop_duplicates(subset=compare_columns).reset_index(drop=True)
            if not existing_df.empty
            else pd.DataFrame(columns=compare_columns)
        )

        incoming_grouped = cls._group_realtime_start_max(incoming)
        existing_grouped = cls._group_realtime_start_max(existing)
        if incoming_grouped.empty:
            return pd.DataFrame(columns=incoming.columns), set(), set()

        grouped = incoming_grouped.merge(
            existing_grouped,
            how="left",
            on=["TICKER_KEY", "OBSERVATION_DATE_KEY"],
            suffixes=("_INCOMING", "_EXISTING"),
        )

        groups_to_replace: set[tuple[str, str]] = set()
        groups_to_insert: set[tuple[str, str]] = set()
        for _, row in grouped.iterrows():
            group_key = (row["TICKER_KEY"], row["OBSERVATION_DATE_KEY"])
            existing_max = row["MAX_REALTIME_START_EXISTING"]
            incoming_max = row["MAX_REALTIME_START_INCOMING"]
            if pd.isna(existing_max):
                groups_to_insert.add(group_key)
                continue
            if incoming_max > existing_max:
                groups_to_replace.add(group_key)

        target_groups = groups_to_replace.union(groups_to_insert)
        if not target_groups:
            return pd.DataFrame(columns=incoming.columns), set(), set()

        keyed_incoming = incoming.copy()
        keyed_incoming["TICKER_KEY"] = (
            keyed_incoming["TICKER"].astype(str).str.strip().str.upper()
        )
        keyed_incoming["OBSERVATION_DATE_KEY"] = keyed_incoming[
            "OBSERVATION_DATE"
        ].apply(lambda value: value.isoformat() if pd.notna(value) else None)
        rows_to_write = incoming.loc[
            keyed_incoming[["TICKER_KEY", "OBSERVATION_DATE_KEY"]].apply(
                lambda row: (row["TICKER_KEY"], row["OBSERVATION_DATE_KEY"])
                in target_groups,
                axis=1,
            )
        ].reset_index(drop=True)
        return rows_to_write, groups_to_replace, groups_to_insert

    def _build_revised_data_group_where_clause(
        self,
        *,
        groups: set[tuple[str, str]],
    ) -> str:
        """Build DELETE predicate for `(TICKER, OBSERVATION_DATE)` groups."""
        if not groups:
            raise ValueError("groups must be non-empty for revised-data delete.")

        grouped_dates: dict[str, list[str]] = {}
        for ticker, observation_date in groups:
            grouped_dates.setdefault(ticker, []).append(observation_date)

        clauses: list[str] = []
        for ticker, observation_dates in grouped_dates.items():
            escaped_ticker = ticker.replace("'", "''")
            escaped_dates = [value.replace("'", "''") for value in observation_dates]
            joined_dates = "', '".join(sorted(set(escaped_dates)))
            clauses.append(
                f"(TICKER = '{escaped_ticker}' "
                f"AND OBSERVATION_DATE IN ('{joined_dates}'))"
            )
        return " OR ".join(clauses)

    def _write_metadata(
        self,
        *,
        engine,
        metadata_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Write changed series metadata rows with series-level upsert semantics."""
        if metadata_df.empty:
            return metadata_df

        tickers = metadata_df["TICKER"].dropna().astype(str).tolist()
        existing_df = self._load_existing_rows(
            engine=engine,
            table_name=self.table_name_metadata,
            tickers=tickers,
        )
        compare_columns = list(metadata_df.columns)
        rows_to_write, skipped = self._filter_new_rows(
            metadata_df,
            existing_df,
            compare_columns=compare_columns,
        )
        log(
            f"FRED metadata diff check: {len(rows_to_write)} inserts/updates, "
            f"{skipped} skipped duplicates."
        )
        if rows_to_write.empty:
            return rows_to_write

        if self._table_exists(engine=engine, table_name=self.table_name_metadata):
            self._delete_rows(
                engine=engine,
                table_name=self.table_name_metadata,
                where_clause=self._build_metadata_where_clause(
                    rows_to_write["TICKER"].dropna().astype(str).tolist()
                ),
            )

        self.azure_data_source.write_sql_table(
            engine=engine,
            table_name=self.table_name_metadata,
            overwrite=False,
            df=rows_to_write,
            dtype_overrides={
                "OBSERVATION_START": satypes.Date(),
                "IS_ACTIVE": satypes.Boolean(),
            },
        )
        return rows_to_write

    def _write_revised_data(
        self,
        *,
        engine,
        revised_data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Replace only groups with a newer max REALTIME_START."""
        if revised_data_df.empty:
            return revised_data_df

        revised_data_df = self._normalize_revised_data(revised_data_df)
        tickers = revised_data_df["TICKER"].dropna().astype(str).unique().tolist()
        existing_df = self._load_existing_rows(
            engine=engine,
            table_name=self.table_name_revised_data,
            tickers=tickers,
        )
        if not existing_df.empty:
            existing_df = self._normalize_revised_data(existing_df)
            existing_df = self._filter_existing_to_incoming_groups(
                existing_df=existing_df,
                incoming_df=revised_data_df,
            )

        rows_to_write, groups_to_replace, groups_to_insert = (
            self._compute_revised_data_group_refreshes(
                revised_data_df,
                existing_df,
            )
        )
        if rows_to_write.empty:
            log("FRED revised-data diff check: no group-level refreshes detected.")
            return pd.DataFrame(columns=revised_data_df.columns)

        if groups_to_replace and self._table_exists(
            engine=engine, table_name=self.table_name_revised_data
        ):
            where_clause = self._build_revised_data_group_where_clause(
                groups=groups_to_replace,
            )
            self._delete_rows(
                engine=engine,
                table_name=self.table_name_revised_data,
                where_clause=where_clause,
            )
            log(
                "FRED revised-data group refresh: deleted "
                f"{len(groups_to_replace)} existing groups with newer incoming vintages."
            )

        log(
            "FRED revised-data group refresh: "
            f"writing {len(rows_to_write)} rows across "
            f"{len(groups_to_replace)} replaced groups and "
            f"{len(groups_to_insert)} new groups."
        )

        self.azure_data_source.write_sql_table(
            engine=engine,
            table_name=self.table_name_revised_data,
            overwrite=False,
            df=rows_to_write,
            dtype_overrides={
                "OBSERVATION_DATE": satypes.Date(),
                "REALTIME_START": satypes.Date(),
                "REALTIME_END": satypes.Date(),
            },
            validation_date_column="OBSERVATION_DATE",
        )
        return rows_to_write

    def refresh_metadata(
        self,
        *,
        configs_path: str | None = None,
        write_to_azure: bool = False,
    ) -> pd.DataFrame:
        """Refresh configured FRED series metadata only."""
        config_df = self._get_series_config_dataframe(configs_path=configs_path)
        client = self._resolve_client(configs_path=configs_path)
        metadata_df = client.get_many_series_metadata(config_df["TICKER"].tolist())
        merged = self._merge_series_metadata(metadata_df, config_df)
        self._update_metadata_cache(merged)
        if write_to_azure:
            engine = self.azure_data_source.get_engine(configs_path=configs_path)
            return self._write_metadata(engine=engine, metadata_df=merged)
        return merged

    def run(
        self,
        *,
        full_history: bool = False,
        observation_start: str | None = None,
        observation_end: str | None = None,
        lookback_days: int | None = None,
        write_to_azure: bool = False,
        configs_path: str | None = None,
    ) -> pd.DataFrame:
        """Load FRED economic data.

        Daily series are loaded as latest-only history across their full
        observation range. Weekly-and-above series are loaded revision-aware,
        filtered by observation date.
        """
        config_df = self._get_series_config_dataframe(configs_path=configs_path)
        client = self._resolve_client(configs_path=configs_path)
        tickers = config_df["TICKER"].tolist()
        metadata_df = self._load_run_metadata(config_df=config_df, client=client)
        daily_tickers, revised_tickers = self._split_tickers_by_frequency(
            tickers=tickers,
            metadata_df=metadata_df,
        )
        fred_cfg = self._load_fred_settings(configs_path=configs_path)
        batch_years = int(
            fred_cfg.get("full_history_batch_years", self.DEFAULT_BATCH_YEARS)
        )
        base_series_options = self._series_options_from_config(config_df)
        run_observation_start = self._resolve_observation_start(
            full_history=full_history,
            configs_path=configs_path,
            start_date=observation_start,
            lookback_days=lookback_days,
        )
        observation_run_end = self._resolve_observation_end(observation_end)
        if full_history:
            run_observation_start = self._resolve_full_history_start(
                metadata_df=metadata_df.loc[
                    metadata_df["TICKER"].isin(revised_tickers)
                ],
                start_date=run_observation_start,
            )
        log(
            "Running FRED pipeline: "
            f"{len(tickers)} series, "
            f"daily_latest_only={len(daily_tickers)}, "
            f"revised={len(revised_tickers)}, "
            f"full_history={full_history}, "
            f"observation_start={run_observation_start}, "
            f"observation_end={observation_run_end}, "
            f"write_to_azure={write_to_azure}"
        )
        engine = (
            self.azure_data_source.get_engine(configs_path=configs_path)
            if write_to_azure
            else None
        )
        frames: list[pd.DataFrame] = []
        observation_start_lookup = self._metadata_observation_start_lookup(metadata_df)
        if revised_tickers and run_observation_start is not None:
            for ticker_number, ticker in enumerate(revised_tickers, start=1):
                series_start = observation_start_lookup.get(
                    ticker,
                    pd.Timestamp(run_observation_start).date(),
                ).isoformat()
                series_observation_start = max(
                    pd.Timestamp(run_observation_start).date(),
                    pd.Timestamp(series_start).date(),
                ).isoformat()
                revised_data_df = self._load_ticker_batches(
                    client=client,
                    ticker=ticker,
                    series_options=base_series_options.get(ticker, {}),
                    observation_start=series_observation_start,
                    observation_end=observation_run_end,
                    ticker_number=ticker_number,
                    ticker_count=len(revised_tickers),
                    batch_years=batch_years,
                    include_revisions=True,
                    mode_label="revised",
                )
                revised_data_df = self._normalize_revised_data(revised_data_df)
                log(
                    "FRED rows fetched: "
                    f"ticker={ticker} "
                    f"mode=revised "
                    f"rows={len(revised_data_df)}"
                )
                if not write_to_azure:
                    frames.append(revised_data_df)
                else:
                    frames.append(
                        self._write_revised_data(
                            engine=engine,
                            revised_data_df=revised_data_df,
                        )
                    )
        for ticker_number, ticker in enumerate(daily_tickers, start=1):
            series_start = observation_start_lookup.get(ticker)
            if series_start is None:
                continue
            daily_data_df = self._load_ticker_batches(
                client=client,
                ticker=ticker,
                series_options=base_series_options.get(ticker, {}),
                observation_start=series_start.isoformat(),
                observation_end=observation_run_end,
                ticker_number=ticker_number,
                ticker_count=len(daily_tickers),
                batch_years=batch_years,
                include_revisions=False,
                mode_label="daily",
            )
            daily_data_df = self._normalize_revised_data(daily_data_df)
            daily_data_df = self._normalize_daily_latest_data(daily_data_df)
            log(
                "FRED rows fetched: "
                f"ticker={ticker} "
                f"mode=daily_latest_only "
                f"rows={len(daily_data_df)}"
            )
            if not write_to_azure:
                frames.append(daily_data_df)
            else:
                frames.append(
                    self._write_revised_data(
                        engine=engine,
                        revised_data_df=daily_data_df,
                    )
                )

        out = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(
                columns=[
                    "TICKER",
                    "OBSERVATION_DATE",
                    "VALUE",
                    "REALTIME_START",
                    "REALTIME_END",
                ]
            )
        )
        if not out.empty:
            out = out.drop_duplicates().reset_index(drop=True)
        log(f"FRED write summary: revised_data={len(out)}")
        return out

    def load_series(self, series_code: str, **kwargs) -> pd.DataFrame:
        """Load one specific series using the configured multi-table workflow."""
        original_codes = self.series_codes
        self.series_codes = [str(series_code).strip().upper()]
        try:
            return self.run(**kwargs)
        finally:
            self.series_codes = original_codes


__all__ = ["FredData"]
