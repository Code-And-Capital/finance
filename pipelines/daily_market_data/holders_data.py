"""Pipeline components for pulling Yahoo holder datasets."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sqlalchemy import types as satypes

import utils.dataframe_utils as dataframe_utils
from pipelines.daily_market_data.yahoo_data import YahooData
from utils.logging import log


class _BaseHolders(YahooData):
    """Shared holder-pipeline behavior for pulling and optional Azure writes."""

    def __init__(
        self,
        *,
        tickers: Iterable[str] | str | None = None,
        yahoo_client=None,
        client=None,
    ) -> None:
        """Initialize shared holder pipeline state."""
        super().__init__(tickers=tickers, yahoo_client=yahoo_client, client=client)
        self.max_yfinance_resets = 10
        self.reset_wait_seconds = 120

    def _pull_holders(
        self,
        *,
        client_method: str,
        date_columns: list[str],
    ) -> pd.DataFrame:
        """Pull one holders dataset with missing-ticker retries."""
        dataframe = self._pull_with_missing_ticker_retries(
            client_method=client_method,
            method_kwargs={},
            date_columns=date_columns,
            max_resets=self.max_yfinance_resets,
            wait_seconds=self.reset_wait_seconds,
        )
        for column in date_columns:
            if column in dataframe.columns:
                dataframe = dataframe_utils.ensure_datetime_column(dataframe, column)
        return dataframe

    def _write_table(
        self,
        *,
        table_name: str,
        dataframe: pd.DataFrame,
        engine,
        date_columns: list[str],
    ) -> None:
        """Write table rows after dedupe-minus-date checks."""
        figi_values = self._extract_figi_values(dataframe)
        existing_df = self._load_existing_rows_for_figi(
            engine=engine,
            table_name=table_name,
            figis=figi_values,
            log_context=table_name,
        )
        rows_to_write = self._filter_new_or_changed_rows(
            incoming_df=dataframe,
            existing_df=existing_df,
            exclude_columns={"DATE"},
        )
        log(
            f"{table_name} rows eligible for write after diff check: "
            f"{len(rows_to_write)}/{len(dataframe)}"
        )
        if rows_to_write.empty:
            log(f"Skipped {table_name} write: no new or changed rows detected.")
            return

        rows_to_write = rows_to_write.copy()
        dtype_overrides: dict[str, satypes.TypeEngine] = {}
        for column in date_columns:
            if column in rows_to_write.columns:
                rows_to_write[column] = (
                    pd.to_datetime(rows_to_write[column], errors="coerce", utc=True)
                    .dt.tz_localize(None)
                    .dt.date
                )
                dtype_overrides[column] = satypes.Date()

        self.azure_data_source.write_sql_table(
            table_name=table_name,
            engine=engine,
            df=rows_to_write,
            overwrite=False,
            dtype_overrides=dtype_overrides,
        )
        log(f"Wrote {table_name} rows to Azure: {len(rows_to_write)}")


class InstitutionalHolders(_BaseHolders):
    """Pull institutional holder snapshots and optionally write to Azure."""

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
        ticker_to_figi: dict[str, str | None] | None = None,
    ) -> pd.DataFrame:
        """Run institutional holders workflow and return pulled rows."""
        log(
            "Running institutional holders pipeline: "
            f"{len(self.tickers)} tickers, write_to_azure={write_to_azure}"
        )
        institutional_holders = self._pull_holders(
            client_method="get_institutional_holders",
            date_columns=["DATE", "DATE_REPORTED"],
        )
        if "VALUE" in institutional_holders.columns:
            institutional_holders = institutional_holders.drop(columns=["VALUE"])
            log("Dropped VALUE column from institutional holders payload")
        institutional_holders = self._attach_figi_from_mapping(
            institutional_holders, ticker_to_figi
        )
        log(f"Pulled institutional holders rows: {len(institutional_holders)}")

        if write_to_azure:
            engine = self.azure_data_source.get_engine(configs_path=configs_path)
            self._write_table(
                table_name="institutional_holders",
                dataframe=institutional_holders,
                engine=engine,
                date_columns=["DATE", "DATE_REPORTED"],
            )

        return institutional_holders


class MajorHolders(_BaseHolders):
    """Pull major holder snapshots and optionally write to Azure."""

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
        ticker_to_figi: dict[str, str | None] | None = None,
    ) -> pd.DataFrame:
        """Run major holders workflow and return pulled rows."""
        log(
            "Running major holders pipeline: "
            f"{len(self.tickers)} tickers, write_to_azure={write_to_azure}"
        )
        major_holders = self._pull_holders(
            client_method="get_major_holders",
            date_columns=["DATE"],
        )
        major_holders = self._attach_figi_from_mapping(major_holders, ticker_to_figi)
        log(f"Pulled major holders rows: {len(major_holders)}")

        if write_to_azure:
            engine = self.azure_data_source.get_engine(configs_path=configs_path)
            self._write_table(
                table_name="major_holders",
                dataframe=major_holders,
                engine=engine,
                date_columns=["DATE"],
            )

        return major_holders
