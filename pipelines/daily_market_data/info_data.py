"""Pipeline component for pulling company info and officers from Yahoo."""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import connectors.yahoo_data_source as yahoo_data_source
import utils.dataframe_utils as dataframe_utils
from utils import geo
from pipelines.daily_market_data.yahoo_data import YahooData
from utils.logging import log


class InfoData(YahooData):
    """Pull company info and officer datasets from one Yahoo client instance."""

    _BASE_INFO_DATE_COLUMNS = [
        "DATE",
        "LASTFISCALYEAREND",
        "NEXTFISCALYEAREND",
        "MOSTRECENTQUARTER",
        "EXDIVIDENDDATE",
        "LASTDIVIDENDDATE",
        "DIVIDENDDATE",
    ]
    _INFO_DATE_COLUMNS = list(
        dict.fromkeys(
            _BASE_INFO_DATE_COLUMNS + list(yahoo_data_source._INFO_EPOCH_DATE_COLUMNS)
        )
    )

    _OFFICER_DATE_COLUMNS = ["DATE"]
    _RESET_WAIT_SECONDS = 120

    def __init__(
        self,
        *,
        tickers=None,
        yahoo_client=None,
        client=None,
        max_workers: int = 10,
    ) -> None:
        """Initialize info pipeline state.

        Parameters
        ----------
        tickers : iterable[str] | str | None
            Ticker symbols to pull from Yahoo.
        yahoo_client : object | None
            Optional injected Yahoo client implementation.
        client : object | None
            Backward-compatible alias for ``yahoo_client``.
        max_workers : int, default 10
            Maximum worker threads when the default Yahoo client is created.
        """
        super().__init__(
            tickers=tickers,
            yahoo_client=yahoo_client,
            client=client,
            max_workers=max_workers,
        )
        self.max_yfinance_resets = 10
        self._latest_info_df: pd.DataFrame | None = None
        self._latest_officers_df: pd.DataFrame | None = None

    def _prepare_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and clean company-info output."""
        out = df.copy()
        out = out.map(lambda value: np.nan if isinstance(value, list) else value)
        for column in self._INFO_DATE_COLUMNS:
            if column in out.columns:
                out = dataframe_utils.ensure_datetime_column(out, column)
        return out

    def _prepare_officers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and clean officers output."""
        out = df.copy()
        for column in self._OFFICER_DATE_COLUMNS:
            if column in out.columns:
                out = dataframe_utils.ensure_datetime_column(out, column)
        return out

    @staticmethod
    def _normalize_ticker_list(tickers) -> list[str]:
        """Normalize ticker symbols to uppercase strings with stable order."""
        return list(
            dict.fromkeys(
                str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()
            )
        )

    @staticmethod
    def _extract_returned_tickers(df: pd.DataFrame) -> set[str]:
        """Extract normalized ticker set from a pull result DataFrame."""
        if "TICKER" not in df.columns or df.empty:
            return set()
        return set(df["TICKER"].dropna().astype(str).str.upper())

    def _pull_remaining(
        self, pending_tickers: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Pull info and officers for all currently remaining tickers."""
        client = self._create_client_for_tickers(pending_tickers)
        info_df = self._prepare_info(client.get_company_info())
        officers_df = self._prepare_officers(client.get_officer_info())
        return info_df, officers_df

    def _pull_info_and_officers(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Pull info/officers in batch retries, creating fresh clients each batch."""
        pending_tickers = self._normalize_ticker_list(self.tickers)
        total_tickers = len(pending_tickers)
        log(f"Starting info/officers pull for {total_tickers} tickers")
        reset_attempts = 0
        info_frames: list[pd.DataFrame] = []
        officer_frames: list[pd.DataFrame] = []

        while pending_tickers:
            info_attempt, officers_attempt = self._pull_remaining(pending_tickers)
            log(
                f"Batch pull returned {len(info_attempt)} info rows and "
                f"{len(officers_attempt)} officer rows"
            )
            if not info_attempt.empty:
                info_frames.append(info_attempt)
            if not officers_attempt.empty:
                officer_frames.append(officers_attempt)

            completed = self._extract_returned_tickers(info_attempt)
            pending_tickers = [
                ticker for ticker in pending_tickers if ticker not in completed
            ]

            if pending_tickers:
                if reset_attempts >= self.max_yfinance_resets:
                    missing = ", ".join(pending_tickers)
                    log(
                        "Reached max yfinance resets "
                        f"({self.max_yfinance_resets}). Missing {len(pending_tickers)} tickers: {missing}",
                        type="warning",
                    )
                    break
                completed_count = total_tickers - len(pending_tickers)
                resume_at = datetime.now() + timedelta(seconds=self._RESET_WAIT_SECONDS)
                log(
                    f"{completed_count}/{total_tickers} tickers completed. "
                    f"Resetting yfinance in {self._RESET_WAIT_SECONDS} seconds "
                    f"(next retry at {resume_at.strftime('%H:%M:%S')})."
                )
                time.sleep(self._RESET_WAIT_SECONDS)
                log("Retrying yfinance now.")
                reset_attempts += 1

        info_df = (
            pd.concat(info_frames, ignore_index=True) if info_frames else pd.DataFrame()
        )
        if "TICKER" in info_df.columns:
            info_df = (
                info_df.assign(TICKER=info_df["TICKER"].astype(str).str.upper())
                .drop_duplicates(subset=["TICKER"], keep="last")
                .reset_index(drop=True)
            )

        officers_df = (
            pd.concat(officer_frames, ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
            if officer_frames
            else pd.DataFrame()
        )
        log(
            f"Final info/officers rows: info={len(info_df)}, officers={len(officers_df)}"
        )
        self._latest_info_df = info_df
        self._latest_officers_df = officers_df
        return info_df, officers_df

    def pull_info(self) -> pd.DataFrame:
        """Return normalized company info using batch-coupled pulls."""
        info_df, _ = self._pull_info_and_officers()
        return info_df

    def pull_officers(self) -> pd.DataFrame:
        """Return normalized company officers using batch-coupled pulls."""
        if self._latest_officers_df is None:
            _, officers_df = self._pull_info_and_officers()
            return officers_df
        return self._latest_officers_df

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
        ticker_to_figi: dict[str, str | None] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return both datasets and optionally write them to Azure SQL."""
        info_df, officers_df = self._pull_info_and_officers()
        info_df = self._attach_figi_from_mapping(info_df, ticker_to_figi)
        officers_df = self._attach_figi_from_mapping(officers_df, ticker_to_figi)
        info_to_write = info_df
        officers_to_write = officers_df
        if write_to_azure:
            engine = self.azure_data_source.get_engine(configs_path=configs_path)
            log(f"Writing company info rows to Azure: {len(info_df)}")
            self.azure_data_source.write_sql_table(
                table_name="company_info",
                engine=engine,
                df=info_df,
                overwrite=False,
            )
            officer_figis = self._extract_figi_values(officers_df)
            existing_officers = self._load_existing_rows_for_figi(
                engine=engine,
                table_name="officers",
                figis=officer_figis,
                log_context="officers",
            )
            officers_to_write = self._filter_new_or_changed_rows(
                incoming_df=officers_df,
                existing_df=existing_officers,
                exclude_columns={"DATE"},
            )
            log(
                f"Officer rows eligible for write after diff check: "
                f"{len(officers_to_write)}/{len(officers_df)}"
            )
            if not officers_to_write.empty:
                self.azure_data_source.write_sql_table(
                    table_name="officers",
                    engine=engine,
                    df=officers_to_write,
                    overwrite=False,
                )
                log(f"Wrote officer rows to Azure: {len(officers_to_write)}")
            else:
                log("Skipped officer write: no new or changed rows detected.")
            address_query = self.sql_client.build_select_all_query(table_name="address")
            try:
                address_df = self.azure_data_source.read_sql_table(
                    engine=engine, query=address_query
                )
                log(f"Loaded cached address rows from Azure: {len(address_df)}")
            except Exception:  # noqa: BLE001
                address_df = pd.DataFrame(
                    columns=["ADDRESS1", "CITY", "COUNTRY", "LAT", "LON"]
                )
                log("Address table read failed, using empty cache", type="warning")

            if {"ADDRESS1", "CITY", "COUNTRY"}.issubset(info_df.columns):
                existing_addresses = (
                    address_df["ADDRESS1"].dropna().unique()
                    if "ADDRESS1" in address_df.columns
                    else []
                )
                missing_address = info_df[
                    ~info_df["ADDRESS1"].isin(existing_addresses)
                ].dropna(subset=["ADDRESS1"])
                log(
                    f"Geocoding {len(missing_address)} missing addresses for address table sync."
                )
                if not missing_address.empty:
                    geocoded = geo.geocode_dataframe(
                        missing_address,
                        cache_df=address_df,
                        delay=0.5,
                    )
                    address_payload = geocoded[
                        ["ADDRESS1", "CITY", "COUNTRY", "LAT", "LON"]
                    ].dropna(subset=["LAT"])
                    log(
                        f"Geocoding produced {len(address_payload)} address rows with coordinates"
                    )
                    if not address_payload.empty:
                        self.azure_data_source.write_sql_table(
                            table_name="address",
                            engine=engine,
                            df=address_payload,
                            overwrite=False,
                        )
                        log(f"Wrote address rows to Azure: {len(address_payload)}")
        return info_to_write, officers_to_write
