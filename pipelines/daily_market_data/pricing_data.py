"""Pipeline component for pulling daily pricing data from Yahoo."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pandas as pd
from pandas.tseries.offsets import BDay
from sqlalchemy import types as satypes

import utils.dataframe_utils as dataframe_utils
from connectors.azure_data_source import default_azure_data_source
from pipelines.daily_market_data.yahoo_data import YahooData
from sql.script_factory import SQLClient
from utils.logging import log


class PricingData(YahooData):
    """Pull pricing data from Yahoo based on max dates stored in Azure SQL."""

    def __init__(
        self,
        *,
        tickers: Iterable[str] | str | None = None,
        yahoo_client=None,
        client=None,
    ) -> None:
        """Initialize the pricing data pipeline component.

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
        self.sql_client = SQLClient()
        self.azure_data_source = default_azure_data_source
        self.table_name = "prices"
        self._today = date.today

    def _build_max_date_query(self) -> str:
        """Build SQL query for ticker-wise max available pricing dates."""
        return self.sql_client.render_sql_query(
            query_path=self.sql_client.resolve_sql_path("max_date.txt"),
            filters={
                "ticker_filter": self.sql_client.add_in_filter("TICKER", self.tickers),
                "table_name": self.table_name,
            },
        )

    def _fetch_max_dates(self, *, configs_path: str | None = None) -> pd.DataFrame:
        """Fetch max dates per ticker from Azure SQL."""
        query = self._build_max_date_query()
        engine = self.azure_data_source.get_engine(configs_path=configs_path)
        max_dates = self.azure_data_source.read_sql_table(engine=engine, query=query)
        log(f"Fetched max-date mapping from Azure: {len(max_dates)} rows")
        return max_dates

    def _build_start_date_mapping(
        self, max_dates: pd.DataFrame, yahoo_client
    ) -> dict[str, pd.Timestamp]:
        """Build ticker->start_date mapping using DB max date + one business day."""
        max_dates = max_dates.copy()
        max_dates["START_DATE"] = pd.to_datetime(max_dates["START_DATE"]) + BDay(1)

        return dataframe_utils.df_to_dict(
            self._add_missing_tickers(max_dates, yahoo_client.tickers),
            "TICKER",
            "START_DATE",
        )

    def _filter_future_start_dates(
        self,
        start_date_mapping: dict[str, pd.Timestamp],
    ) -> tuple[dict[str, pd.Timestamp], list[str]]:
        """Return eligible start dates and skipped tickers beyond today's date."""
        today = self._today()
        eligible: dict[str, pd.Timestamp] = {}
        skipped: list[str] = []

        for ticker, start_date in start_date_mapping.items():
            start_day = pd.Timestamp(start_date).date()
            if start_day > today:
                skipped.append(str(ticker))
            else:
                eligible[str(ticker)] = start_date

        return eligible, skipped

    @staticmethod
    def _add_missing_tickers(
        df: pd.DataFrame,
        ticker_list: list[str],
    ) -> pd.DataFrame:
        """Ensure all tickers exist in the DataFrame with a default START_DATE."""
        if "TICKER" not in df.columns:
            raise KeyError("DataFrame must contain a 'TICKER' column")

        out = df.copy()
        existing = set(out["TICKER"])
        missing = [ticker for ticker in ticker_list if ticker not in existing]

        if missing:
            new_rows = pd.DataFrame({"TICKER": missing, "START_DATE": "2000-01-01"})
            out = pd.concat([out, new_rows], ignore_index=True)

        return out

    @staticmethod
    def _coerce_date_only(df: pd.DataFrame, column: str = "DATE") -> pd.DataFrame:
        """Convert a datetime column to python date objects."""
        out = df.copy()
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="raise").dt.date
        return out

    @staticmethod
    def _find_adjusted_tickers(df: pd.DataFrame) -> list[str]:
        """Return tickers with dividends or stock-split corporate actions."""
        required = {"TICKER", "DIVIDENDS", "STOCK_SPLITS"}
        if not required.issubset(df.columns):
            return []

        adjusted = df[(df["DIVIDENDS"] > 0) | (df["STOCK_SPLITS"] > 0)][
            "TICKER"
        ].unique()
        return [str(ticker) for ticker in adjusted]

    def _pull_adjusted_prices(
        self,
        adjusted_tickers: list[str],
        *,
        use_start_date_mapping: bool,
        configs_path: str | None,
    ) -> pd.DataFrame:
        """Re-pull pricing history for adjusted tickers only."""
        return PricingData(tickers=adjusted_tickers).run(
            use_start_date_mapping=use_start_date_mapping,
            write_to_azure=False,
            adjust_for_corporate_actions=False,
            configs_path=configs_path,
        )

    def _overwrite_adjusted_tickers(
        self,
        *,
        engine,
        adjusted_tickers: list[str],
        use_start_date_mapping: bool,
        configs_path: str | None,
    ) -> None:
        """Delete and reload full history for adjusted tickers."""
        log(f"Overwriting adjusted tickers in Azure: {len(adjusted_tickers)} tickers")
        escaped = [ticker.replace("'", "''") for ticker in adjusted_tickers]
        joined = "', '".join(escaped)
        where_clause = f"TICKER IN ('{joined}')"
        delete_query = self.sql_client.build_delete_query(
            table_name=self.table_name,
            where_clause=where_clause,
            schema="dbo",
        )

        self.azure_data_source.delete_sql_rows(query=delete_query, engine=engine)
        log("Deleted existing price rows for adjusted tickers")
        adjusted_prices = self._pull_adjusted_prices(
            adjusted_tickers,
            use_start_date_mapping=use_start_date_mapping,
            configs_path=configs_path,
        )
        log(f"Re-pulled adjusted price rows: {len(adjusted_prices)}")
        self.azure_data_source.write_sql_table(
            engine=engine,
            table_name=self.table_name,
            overwrite=False,
            df=adjusted_prices,
        )
        log(f"Wrote adjusted price rows to Azure: {len(adjusted_prices)}")

    def run(
        self,
        *,
        use_start_date_mapping: bool = False,
        write_to_azure: bool = False,
        adjust_for_corporate_actions: bool = True,
        configs_path: str | None = None,
    ) -> pd.DataFrame:
        """Execute the end-to-end pricing pull workflow."""
        log(
            "Running pricing pipeline: "
            f"{len(self.tickers)} tickers, "
            f"use_start_date_mapping={use_start_date_mapping}, "
            f"write_to_azure={write_to_azure}"
        )
        start_date_mapping: dict[str, pd.Timestamp] | None = None
        expected_tickers: list[str] = [
            str(ticker).strip().upper() for ticker in self.tickers
        ]
        if use_start_date_mapping:
            yahoo_client = self._resolve_client()
            max_dates = self._fetch_max_dates(configs_path=configs_path)
            start_date_mapping = self._build_start_date_mapping(max_dates, yahoo_client)
            log(f"Built start-date mapping for {len(start_date_mapping)} tickers")
            eligible_mapping, skipped_tickers = self._filter_future_start_dates(
                start_date_mapping
            )
            expected_tickers = [
                str(ticker).strip().upper() for ticker in eligible_mapping.keys()
            ]
            if skipped_tickers:
                log(
                    "Skipping tickers with start_date after today: "
                    f"{len(skipped_tickers)} -> {skipped_tickers}",
                    type="warning",
                )
            if not eligible_mapping:
                log(
                    "No tickers left after start-date filtering; returning empty pricing dataframe."
                )
                dataframe = pd.DataFrame()
            else:
                original_client = self.yahoo_client
                pull_client = yahoo_client
                if len(eligible_mapping) != len(start_date_mapping):
                    pull_client = self._create_client_for_tickers(
                        list(eligible_mapping.keys())
                    )
                self.yahoo_client = pull_client
                try:
                    dataframe = self._pull_generic(
                        client_method="get_prices",
                        method_kwargs={"start_date": eligible_mapping},
                        date_columns=["DATE"],
                    )
                finally:
                    self.yahoo_client = original_client
        else:
            dataframe = self._pull_generic(
                client_method="get_prices",
                method_kwargs={},
                date_columns=["DATE"],
            )
        returned_tickers = set(
            dataframe["TICKER"].dropna().astype(str).str.upper()
            if "TICKER" in dataframe.columns
            else []
        )
        missing_tickers = [
            ticker
            for ticker in expected_tickers
            if ticker and ticker not in returned_tickers
        ]
        for ticker in missing_tickers:
            log(
                f"No pricing data returned for ticker: {ticker}",
                type="warning",
            )
        log(f"Pulled pricing rows: {len(dataframe)}")
        if not dataframe.empty:
            dataframe = dataframe_utils.ensure_datetime_column(dataframe, "DATE")
            dataframe = self._coerce_date_only(dataframe, "DATE")
        else:
            log("Pricing pull returned an empty dataframe", type="warning")

        if write_to_azure:
            engine = self.azure_data_source.get_engine(configs_path=configs_path)
            self.azure_data_source.write_sql_table(
                engine=engine,
                table_name="prices",
                overwrite=False,
                df=dataframe,
            )
            log(f"Wrote pricing rows to Azure: {len(dataframe)}")
            if adjust_for_corporate_actions:
                adjusted_tickers = self._find_adjusted_tickers(dataframe)
                if adjusted_tickers and start_date_mapping is not None:
                    cutoff = pd.Timestamp("2000-01-01")
                    existing_tickers = {
                        ticker
                        for ticker, mapped_start in start_date_mapping.items()
                        if pd.Timestamp(mapped_start) > cutoff
                    }
                    before_count = len(adjusted_tickers)
                    adjusted_tickers = [
                        ticker
                        for ticker in adjusted_tickers
                        if ticker in existing_tickers
                    ]
                    if before_count != len(adjusted_tickers):
                        log(
                            "Filtered adjusted tickers to DB-existing names only: "
                            f"{len(adjusted_tickers)}/{before_count}",
                        )
                if adjusted_tickers:
                    log(
                        f"Adjusted tickers identified for full overwrite: {adjusted_tickers}"
                    )
                    self._overwrite_adjusted_tickers(
                        engine=engine,
                        adjusted_tickers=adjusted_tickers,
                        use_start_date_mapping=use_start_date_mapping,
                        configs_path=configs_path,
                    )

        return dataframe


class AnalystPriceTargetsData(YahooData):
    """Pull analyst price targets and optionally write them to Azure."""

    def __init__(
        self,
        *,
        tickers: Iterable[str] | str | None = None,
        yahoo_client=None,
        client=None,
    ) -> None:
        """Initialize analyst price targets pipeline state."""
        super().__init__(tickers=tickers, yahoo_client=yahoo_client, client=client)
        self.table_name = "analyst_price_targets"
        self.max_yfinance_resets = 10
        self.reset_wait_seconds = 20

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
    ) -> pd.DataFrame:
        """Run analyst price targets pull workflow with missing-ticker retries."""
        log(
            "Running analyst price targets pipeline: "
            f"{len(self.tickers)} tickers, write_to_azure={write_to_azure}"
        )
        dataframe = self._pull_with_missing_ticker_retries(
            client_method="get_analyst_price_targets",
            method_kwargs={},
            date_columns=["DATE"],
            max_resets=self.max_yfinance_resets,
            wait_seconds=self.reset_wait_seconds,
        )
        if "CURRENT" in dataframe.columns:
            dataframe = dataframe.drop(columns=["CURRENT"])
            log("Dropped CURRENT column from analyst price target payload")
        if "DATE" in dataframe.columns:
            dataframe = dataframe_utils.ensure_datetime_column(dataframe, "DATE")
        log(f"Pulled analyst price target rows: {len(dataframe)}")

        if write_to_azure:
            engine = self.azure_data_source.get_engine(configs_path=configs_path)
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
                "Analyst price target rows eligible for write after diff check: "
                f"{len(rows_to_write)}/{len(dataframe)}"
            )
            if rows_to_write.empty:
                log(
                    "Skipped analyst price target write: no new or changed rows detected."
                )
            else:
                rows_to_write = rows_to_write.copy()
                rows_to_write["DATE"] = (
                    pd.to_datetime(rows_to_write["DATE"], errors="coerce", utc=True)
                    .dt.tz_localize(None)
                    .dt.date
                )
                self.azure_data_source.write_sql_table(
                    engine=engine,
                    table_name=self.table_name,
                    overwrite=False,
                    df=rows_to_write,
                    dtype_overrides={"DATE": satypes.Date()},
                )
                log(f"Wrote analyst price target rows to Azure: {len(rows_to_write)}")

        return dataframe
