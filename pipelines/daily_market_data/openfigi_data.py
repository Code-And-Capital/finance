"""Pipeline component for enriching securities with OpenFIGI mappings."""

from typing import Any

import pandas as pd

from connectors.azure_data_source import default_azure_data_source
from connectors.open_figi_data_source import OpenFigiDataSource
from sql.script_factory import SQLClient
from utils.logging import log


class OpenFigiData:
    """Orchestrate OpenFIGI enrichment and optional Azure writes."""

    def __init__(
        self,
        *,
        universe_df: pd.DataFrame,
        openfigi_client: OpenFigiDataSource | None = None,
        client: OpenFigiDataSource | None = None,
        openfigi_api_key: str | None = None,
    ) -> None:
        """Initialize OpenFIGI pipeline state.

        Parameters
        ----------
        universe_df
            Dataframe with required columns: ``TICKER``, ``NAME``, ``LOCATION``.
        openfigi_client
            Optional injected ``OpenFigiDataSource`` instance.
        client
            Backward-compatible alias for ``openfigi_client``.
        openfigi_api_key
            Optional API key used when creating an internal client.
        """
        if openfigi_client is not None and client is not None:
            raise ValueError("Pass only one of 'openfigi_client' or 'client'.")

        self.universe_df = universe_df.copy()
        self.openfigi_client = (
            openfigi_client if openfigi_client is not None else client
        )
        self.openfigi_api_key = openfigi_api_key
        self.clients_used: list[OpenFigiDataSource] = []
        self.azure_data_source = default_azure_data_source
        self.sql_client = SQLClient()
        self.table_name = "security_master"

    def _track_client(self, client: OpenFigiDataSource) -> None:
        """Track a client instance once for lifecycle introspection."""
        if not any(existing is client for existing in self.clients_used):
            self.clients_used.append(client)

    def _create_client(self) -> OpenFigiDataSource:
        """Create an OpenFIGI data-source client for this pipeline run."""
        client = OpenFigiDataSource(
            universe_df=self.universe_df,
            api_key=self.openfigi_api_key,
        )
        self._track_client(client)
        return client

    def _resolve_client(self) -> OpenFigiDataSource:
        """Resolve an injected client or create a default one."""
        if self.openfigi_client is not None:
            self._track_client(self.openfigi_client)
            return self.openfigi_client
        self.openfigi_client = self._create_client()
        return self.openfigi_client

    def _pull_generic(
        self,
        client_method: str,
        method_kwargs: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Invoke a client pull method by name."""
        client = self._resolve_client()
        method = getattr(client, client_method)
        return method(**(method_kwargs or {}))

    @staticmethod
    def _normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize key columns used for universe-difference checks."""
        out = df.copy()
        out["TICKER"] = out["TICKER"].astype("string").str.strip().str.upper()
        out["NAME"] = out["NAME"].astype("string").str.strip()
        out = out.dropna(subset=["TICKER", "NAME"])
        out = out[(out["TICKER"] != "") & (out["NAME"] != "")]
        return out[["TICKER", "NAME"]].drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _is_us_location(location: object) -> bool:
        """Return True when location maps to the United States."""
        if location is None or pd.isna(location):
            return False
        text = str(location).strip().lower()
        return text in {"united states", "united states of america", "usa", "us"}

    def _filter_existing_universe(self, *, configs_path: str | None) -> pd.DataFrame:
        """Return only rows not present in Azure based on US/international keys."""
        source_df = self.universe_df.loc[
            :, ~self.universe_df.columns.duplicated()
        ].copy()
        required = {"TICKER", "NAME", "LOCATION"}
        missing = sorted(required.difference(source_df.columns))
        if missing:
            raise ValueError(
                "OpenFigiData requires universe_df columns: "
                f"TICKER, NAME, LOCATION. Missing: {missing}"
            )

        incoming_full = source_df[["TICKER", "NAME", "LOCATION"]].copy()
        incoming_full["TICKER"] = (
            incoming_full["TICKER"].astype("string").str.strip().str.upper()
        )
        incoming_full["NAME"] = incoming_full["NAME"].astype("string").str.strip()
        incoming_full["LOCATION"] = (
            incoming_full["LOCATION"].astype("string").str.strip()
        )
        incoming_full = incoming_full.dropna(subset=["TICKER"])
        incoming_full = incoming_full[incoming_full["TICKER"] != ""]
        incoming_full["IS_US"] = incoming_full["LOCATION"].apply(self._is_us_location)
        if incoming_full.empty:
            return incoming_full.iloc[0:0]

        try:
            engine = self.azure_data_source.get_engine(configs_path=configs_path)
            query = self.sql_client.build_select_all_query(table_name=self.table_name)
            existing_raw = self.azure_data_source.read_sql_table(
                engine=engine, query=query
            )
        except Exception as exc:  # noqa: BLE001
            log(
                f"Could not load existing security master rows for filtering: {exc}. "
                "Proceeding with full OpenFIGI universe.",
                type="warning",
            )
            return incoming_full

        key_required = {"TICKER", "NAME"}
        country_col = None
        if "COUNTRY" in existing_raw.columns:
            country_col = "COUNTRY"
        elif "LOCATION" in existing_raw.columns:
            country_col = "LOCATION"

        if existing_raw.empty or not key_required.issubset(existing_raw.columns):
            return incoming_full

        existing = existing_raw.copy()
        existing["TICKER"] = existing["TICKER"].astype("string").str.strip().str.upper()
        existing["NAME"] = existing["NAME"].astype("string").str.strip()
        if country_col is not None:
            existing["COUNTRY_KEY"] = existing[country_col].astype("string").str.strip()
        else:
            existing["COUNTRY_KEY"] = pd.NA

        us_incoming = incoming_full[incoming_full["IS_US"]].copy()
        intl_incoming = incoming_full[~incoming_full["IS_US"]].copy()

        us_to_run = us_incoming
        if not us_incoming.empty:
            us_existing_keys = (
                existing[["TICKER", "NAME"]]
                .dropna(subset=["TICKER", "NAME"])
                .drop_duplicates()
            )
            us_missing = us_incoming.merge(
                us_existing_keys,
                on=["TICKER", "NAME"],
                how="left",
                indicator=True,
            )
            us_to_run = us_missing[us_missing["_merge"] == "left_only"][
                ["TICKER", "NAME", "LOCATION", "IS_US"]
            ]

        intl_to_run = intl_incoming
        if not intl_incoming.empty:
            if country_col is None:
                log(
                    "security_master has no COUNTRY/LOCATION column; cannot perform "
                    "international ticker+country match. Running all international rows.",
                    type="warning",
                )
            else:
                intl_existing_keys = (
                    existing[["TICKER", "COUNTRY_KEY"]]
                    .dropna(subset=["TICKER", "COUNTRY_KEY"])
                    .drop_duplicates()
                )
                intl_missing = intl_incoming.merge(
                    intl_existing_keys,
                    left_on=["TICKER", "LOCATION"],
                    right_on=["TICKER", "COUNTRY_KEY"],
                    how="left",
                    indicator=True,
                )
                intl_to_run = intl_missing[intl_missing["_merge"] == "left_only"][
                    ["TICKER", "NAME", "LOCATION", "IS_US"]
                ]

        to_run = pd.concat([us_to_run, intl_to_run], ignore_index=True)
        if to_run.empty:
            return incoming_full.iloc[0:0]
        return (
            to_run.drop(columns=["IS_US"], errors="ignore")
            .drop_duplicates(subset=["TICKER", "NAME", "LOCATION"])
            .reset_index(drop=True)
        )

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
    ) -> pd.DataFrame:
        """Execute OpenFIGI enrichment and optionally write to Azure."""
        self.universe_df = self._filter_existing_universe(configs_path=configs_path)
        log(
            "Running OpenFIGI pipeline: "
            f"{len(self.universe_df)} universe rows, write_to_azure={write_to_azure}"
        )
        if self.universe_df.empty:
            log("OpenFIGI pipeline skipped: no new TICKER/NAME/LOCATION rows to map.")
            return pd.DataFrame()

        figi_df = self._pull_generic(client_method="get_security_master")
        mapped_count = (
            int(figi_df["FIGI"].notna().sum()) if "FIGI" in figi_df.columns else 0
        )
        log(
            f"OpenFIGI pipeline complete: {len(figi_df)} rows, "
            f"FIGI mapped={mapped_count}"
        )

        if write_to_azure:
            engine = self.azure_data_source.get_engine(configs_path=configs_path)
            self.azure_data_source.write_sql_table(
                table_name=self.table_name,
                engine=engine,
                df=figi_df,
                overwrite=False,
            )
            log(
                f"Wrote OpenFIGI rows to Azure table '{self.table_name}': "
                f"{len(figi_df)}"
            )

        return figi_df
