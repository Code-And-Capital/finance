"""Pipeline component for pulling Yahoo insider transaction data."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sqlalchemy import types as satypes

import utils.dataframe_utils as dataframe_utils
from pipelines.daily_market_data.yahoo_data import YahooData
from utils.logging import log


class InsiderTransactionsData(YahooData):
    """Pull insider transactions from Yahoo and optionally write to Azure."""

    def __init__(
        self,
        *,
        tickers: Iterable[str] | str | None = None,
        yahoo_client=None,
        client=None,
    ) -> None:
        """Initialize insider transactions pipeline state."""
        super().__init__(tickers=tickers, yahoo_client=yahoo_client, client=client)
        self.table_name = "insider_transactions"
        self.max_yfinance_resets = 10
        self.reset_wait_seconds = 120

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
        ticker_to_figi: dict[str, str | None] | None = None,
    ) -> pd.DataFrame:
        """Run insider transactions pull workflow with missing-ticker retries."""
        log(
            "Running insider transactions pipeline: "
            f"{len(self.tickers)} tickers, write_to_azure={write_to_azure}"
        )
        dataframe = self._pull_with_missing_ticker_retries(
            client_method="get_insider_transactions",
            method_kwargs={},
            date_columns=["DATE", "START_DATE"],
            max_resets=self.max_yfinance_resets,
            wait_seconds=self.reset_wait_seconds,
        )
        log(f"Pulled insider transactions rows: {len(dataframe)}")

        for column in ["DATE", "START_DATE"]:
            if column in dataframe.columns:
                dataframe = dataframe_utils.ensure_datetime_column(dataframe, column)
        if "URL" in dataframe.columns:
            normalized_url = dataframe["URL"].astype("string").str.strip()
            dataframe["URL"] = normalized_url.where(normalized_url != "", pd.NA)
        dataframe = self._attach_figi_from_mapping(dataframe, ticker_to_figi)

        if write_to_azure:
            engine = self.azure_data_source.get_engine(configs_path=configs_path)
            figi_values = self._extract_figi_values(dataframe)
            existing_df = self._load_existing_rows_for_figi(
                engine=engine,
                table_name=self.table_name,
                figis=figi_values,
                log_context=self.table_name,
            )
            rows_to_write = self._filter_new_or_changed_rows(
                incoming_df=dataframe,
                existing_df=existing_df,
                exclude_columns={"DATE"},
            )
            log(
                "Insider transactions rows eligible for write after diff check: "
                f"{len(rows_to_write)}/{len(dataframe)}"
            )
            if rows_to_write.empty:
                log(
                    "Skipped insider transactions write: no new or changed rows detected."
                )
            else:
                rows_to_write = rows_to_write.copy()
                if "URL" in rows_to_write.columns:
                    normalized_url = rows_to_write["URL"].astype("string").str.strip()
                    rows_to_write["URL"] = normalized_url.where(
                        normalized_url != "", pd.NA
                    )
                for column in ["DATE", "START_DATE"]:
                    if column in rows_to_write.columns:
                        rows_to_write[column] = (
                            pd.to_datetime(
                                rows_to_write[column], errors="coerce", utc=True
                            )
                            .dt.tz_localize(None)
                            .dt.date
                        )
                self.azure_data_source.write_sql_table(
                    table_name=self.table_name,
                    engine=engine,
                    df=rows_to_write,
                    overwrite=False,
                    dtype_overrides={
                        "DATE": satypes.Date(),
                        "START_DATE": satypes.Date(),
                    },
                )
                log(f"Wrote insider transactions rows to Azure: {len(rows_to_write)}")

        return dataframe
