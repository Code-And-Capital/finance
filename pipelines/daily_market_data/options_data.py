"""Pipeline component for pulling Yahoo options-chain data."""

from collections.abc import Iterable

import pandas as pd
from sqlalchemy import types as satypes

import utils.dataframe_utils as dataframe_utils
from pipelines.daily_market_data.yahoo_data import YahooData
from utils.logging import log


class OptionsData(YahooData):
    """Pull options-chain snapshots from Yahoo and optionally write to Azure."""

    def __init__(
        self,
        *,
        tickers: Iterable[str] | str | None = None,
        yahoo_client=None,
        client=None,
    ) -> None:
        """Initialize the options pipeline component.

        Parameters
        ----------
        tickers
            Ticker symbols to pull from Yahoo.
        yahoo_client
            Optional injected Yahoo client implementation.
        client
            Backward-compatible alias for ``yahoo_client``.
        """
        super().__init__(tickers=tickers, yahoo_client=yahoo_client, client=client)
        self.table_name = "options"
        self.max_yfinance_resets = 10
        self.reset_wait_seconds = 120

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
        ticker_to_figi: dict[str, str | None] | None = None,
    ) -> pd.DataFrame:
        """Run options pull workflow with retry support for missing tickers."""
        log(
            "Running options pipeline: "
            f"{len(self.tickers)} tickers, write_to_azure={write_to_azure}"
        )
        dataframe = self._pull_with_missing_ticker_retries(
            client_method="get_options",
            method_kwargs={},
            date_columns=["DATE", "LASTTRADEDATE", "EXPIRATION"],
            max_resets=self.max_yfinance_resets,
            wait_seconds=self.reset_wait_seconds,
        )
        log(f"Pulled options rows: {len(dataframe)}")

        for column in ["DATE", "LASTTRADEDATE", "EXPIRATION"]:
            if column in dataframe.columns:
                dataframe = dataframe_utils.ensure_datetime_column(dataframe, column)
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
                f"Options rows eligible for write after diff check: "
                f"{len(rows_to_write)}/{len(dataframe)}"
            )
            if rows_to_write.empty:
                log("Skipped options write: no new or changed rows detected.")
                return rows_to_write
            else:
                rows_to_write = rows_to_write.copy()
                for column in ["DATE", "LASTTRADEDATE", "EXPIRATION"]:
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
                        "LASTTRADEDATE": satypes.Date(),
                        "EXPIRATION": satypes.Date(),
                    },
                )
                log(f"Wrote options rows to Azure: {len(rows_to_write)}")
                return rows_to_write

        return dataframe
