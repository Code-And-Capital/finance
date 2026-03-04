"""Pipeline components for Yahoo analyst recommendations and upgrades/downgrades."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sqlalchemy import types as satypes

import utils.dataframe_utils as dataframe_utils
from pipelines.daily_market_data.yahoo_data import YahooData
from utils.logging import log


class _BaseAnalystData(YahooData):
    """Base class for analyst table pulls with shared Azure write behavior."""

    table_name: str = ""
    client_method: str = ""
    date_columns: list[str] = ["DATE"]
    log_label: str = "analyst data"

    def __init__(
        self,
        *,
        tickers: Iterable[str] | str | None = None,
        yahoo_client=None,
        client=None,
    ) -> None:
        """Initialize shared analyst pipeline state."""
        super().__init__(tickers=tickers, yahoo_client=yahoo_client, client=client)
        self.max_yfinance_resets = 10
        self.reset_wait_seconds = 120

    def _pull_data(self) -> pd.DataFrame:
        """Pull one analyst dataset with missing-ticker retries."""
        dataframe = self._pull_with_missing_ticker_retries(
            client_method=self.client_method,
            method_kwargs={},
            date_columns=self.date_columns,
            max_resets=self.max_yfinance_resets,
            wait_seconds=self.reset_wait_seconds,
        )
        for column in self.date_columns:
            if column in dataframe.columns:
                dataframe = dataframe_utils.ensure_datetime_column(dataframe, column)
        return dataframe

    def _write_table(self, *, dataframe: pd.DataFrame, engine) -> None:
        """Write one analyst table after dedupe-minus-date checks."""
        existing_df = self._load_existing_rows_for_tickers(
            engine=engine,
            table_name=self.table_name,
            log_context=self.table_name,
        )
        rows_to_write = self._filter_new_or_changed_rows(
            incoming_df=dataframe,
            existing_df=existing_df,
            exclude_columns={"DATE"},
        )
        log(
            f"{self.table_name} rows eligible for write after diff check: "
            f"{len(rows_to_write)}/{len(dataframe)}"
        )
        if rows_to_write.empty:
            log(f"Skipped {self.table_name} write: no new or changed rows detected.")
            return

        rows_to_write = rows_to_write.copy()
        dtype_overrides: dict[str, satypes.TypeEngine] = {}
        for column in self.date_columns:
            if column in rows_to_write.columns:
                rows_to_write[column] = (
                    pd.to_datetime(rows_to_write[column], errors="coerce", utc=True)
                    .dt.tz_localize(None)
                    .dt.date
                )
                dtype_overrides[column] = satypes.Date()

        self.azure_data_source.write_sql_table(
            table_name=self.table_name,
            engine=engine,
            df=rows_to_write,
            overwrite=False,
            dtype_overrides=dtype_overrides,
        )
        log(f"Wrote {self.table_name} rows to Azure: {len(rows_to_write)}")

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
    ) -> pd.DataFrame:
        """Run analyst dataset workflow and return the resulting DataFrame."""
        log(
            f"Running {self.log_label} pipeline: "
            f"{len(self.tickers)} tickers, write_to_azure={write_to_azure}"
        )
        dataframe = self._pull_data()
        log(f"Pulled {self.log_label} rows: {len(dataframe)}")
        returned_tickers = self._extract_returned_tickers(dataframe)
        missing_tickers = [
            ticker
            for ticker in self._normalize_ticker_list(self.tickers)
            if ticker not in returned_tickers
        ]
        if missing_tickers:
            log(
                f"Could not pull {self.log_label} for {len(missing_tickers)} tickers: "
                f"{', '.join(missing_tickers)}",
                type="warning",
            )

        if write_to_azure:
            engine = self.azure_data_source.get_engine(configs_path=configs_path)
            self._write_table(dataframe=dataframe, engine=engine)

        return dataframe


class AnalystRecommendationsData(_BaseAnalystData):
    """Pull analyst recommendations, optionally writing to Azure."""

    table_name = "analyst_recommendations"
    client_method = "get_recommendations"
    date_columns = ["DATE"]
    log_label = "analyst recommendations"


class AnalystUpgradesDowngradesData(_BaseAnalystData):
    """Pull analyst upgrades/downgrades, optionally writing to Azure."""

    table_name = "analyst_upgrades_downgrades"
    client_method = "get_upgrades_downgrades"
    date_columns = ["DATE", "GRADEDATE"]
    log_label = "analyst upgrades/downgrades"
