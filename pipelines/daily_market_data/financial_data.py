"""Pipeline components for pulling Yahoo financial datasets."""

import time
from collections.abc import Iterable

import pandas as pd
from sqlalchemy import types as satypes

import utils.dataframe_utils as dataframe_utils
from pipelines.daily_market_data.yahoo_data import YahooData
from utils.logging import log


class FinancialData(YahooData):
    """Pull annual and quarterly Yahoo financial statements across statement types."""

    _PAUSE_SECONDS = 120

    _PULL_SEQUENCE: tuple[tuple[str, bool], ...] = (
        ("financial", True),
        ("financial", False),
        ("balance_sheet", True),
        ("balance_sheet", False),
        ("income_statement", True),
        ("income_statement", False),
        ("cashflow", True),
        ("cashflow", False),
    )
    _TABLE_NAME_MAP: dict[str, str] = {
        "financial_annual": "financial_annual",
        "financial_quarterly": "financial_quarterly",
        "balance_sheet_annual": "balancesheet_annual",
        "balance_sheet_quarterly": "balancesheet_quarterly",
        "income_statement_annual": "incomestatement_annual",
        "income_statement_quarterly": "incomestatement_quarterly",
        "cashflow_annual": "cashflow_annual",
        "cashflow_quarterly": "cashflow_quarterly",
    }

    def __init__(
        self,
        *,
        tickers: Iterable[str] | str | None = None,
        yahoo_client=None,
        client=None,
    ) -> None:
        """Initialize the financial data pipeline component."""
        super().__init__(tickers=tickers, yahoo_client=yahoo_client, client=client)
        self.pause_seconds = self._PAUSE_SECONDS
        self._sleep = time.sleep
        self.max_yfinance_resets = 10
        self.reset_wait_seconds = 120

    def _pull_financials(self, *, statement_type: str, annual: bool) -> pd.DataFrame:
        """Pull one financial dataset from Yahoo and log its row count."""
        period = "annual" if annual else "quarterly"
        log(f"Pulling {statement_type} ({period}) for {len(self.tickers)} tickers")
        dataframe = self._pull_with_missing_ticker_retries(
            client_method="get_financials",
            method_kwargs={"statement_type": statement_type, "annual": annual},
            date_columns=["DATE", "REPORT_DATE"],
            max_resets=self.max_yfinance_resets,
            wait_seconds=self.reset_wait_seconds,
        )
        dataframe = self._coerce_date_only(dataframe, "DATE")
        dataframe = self._coerce_date_only(dataframe, "REPORT_DATE")
        log(f"Pulled {statement_type} ({period}) rows: {len(dataframe)}")
        self._warn_missing_tickers(
            dataframe=dataframe, statement_type=statement_type, annual=annual
        )
        return dataframe

    @staticmethod
    def _coerce_date_only(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Coerce datetime-like values to Python date objects for SQL DATE writes."""
        if column not in df.columns or df.empty:
            return df
        out = dataframe_utils.ensure_datetime_column(df, column)
        out[column] = out[column].dt.date
        return out

    def _warn_missing_tickers(
        self,
        *,
        dataframe: pd.DataFrame,
        statement_type: str,
        annual: bool,
    ) -> None:
        """Log warning for tickers missing from a financial statement pull."""
        expected = {
            str(ticker).strip().upper()
            for ticker in self.tickers
            if str(ticker).strip()
        }
        period = "annual" if annual else "quarterly"

        if not expected:
            return

        if dataframe.empty or "TICKER" not in dataframe.columns:
            missing = sorted(expected)
        else:
            returned = set(dataframe["TICKER"].dropna().astype(str).str.upper())
            missing = sorted(expected - returned)

        if missing:
            log(
                f"Missing tickers for {statement_type} ({period}): "
                f"{len(missing)} -> {missing}",
                type="warning",
            )

    @staticmethod
    def _drop_all_null_payload_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where all payload columns are null.

        Key columns ``DATE``, ``TICKER``, and ``REPORT_DATE`` are excluded from
        the null check.
        """
        if dataframe.empty:
            return dataframe

        excluded = {"DATE", "TICKER", "REPORT_DATE"}
        payload_columns = [
            column for column in dataframe.columns if column not in excluded
        ]
        if not payload_columns:
            return dataframe.iloc[0:0].copy()

        mask = dataframe[payload_columns].notna().any(axis=1)
        return dataframe.loc[mask].copy()

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
        ticker_to_figi: dict[str, str | None] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run the full financial pull sequence and return all statement datasets."""
        outputs: dict[str, pd.DataFrame] = {}

        for index, (statement_type, annual) in enumerate(self._PULL_SEQUENCE):
            key = f"{statement_type}_{'annual' if annual else 'quarterly'}"
            outputs[key] = self._pull_financials(
                statement_type=statement_type,
                annual=annual,
            )
            outputs[key] = self._attach_figi_from_mapping(outputs[key], ticker_to_figi)

            has_next = index < len(self._PULL_SEQUENCE) - 1
            if has_next and self.pause_seconds > 0:
                log(f"Sleeping {self.pause_seconds} seconds before next financial pull")
                self._sleep(self.pause_seconds)

        log(
            "Financial data pipeline complete: "
            + ", ".join(f"{name}={len(df)}" for name, df in outputs.items())
        )

        if not write_to_azure:
            return outputs

        write_outputs: dict[str, pd.DataFrame] = {}
        engine = self.azure_data_source.get_engine(configs_path=configs_path)
        for key, dataframe in outputs.items():
            table_name = self._TABLE_NAME_MAP[key]
            filtered_dataframe = self._drop_all_null_payload_rows(dataframe)
            dropped_count = len(dataframe) - len(filtered_dataframe)
            if dropped_count > 0:
                log(
                    f"Dropped {dropped_count} all-null payload rows for '{table_name}' "
                    "(excluding DATE/TICKER/REPORT_DATE)."
                )
            figi_values = self._extract_figi_values(filtered_dataframe)
            existing_df = self._load_existing_rows_for_figi(
                engine=engine,
                table_name=table_name,
                figis=figi_values,
                log_context=table_name,
            )
            rows_to_write = self._filter_new_or_changed_rows(
                incoming_df=filtered_dataframe,
                existing_df=existing_df,
                exclude_columns={"DATE"},
            )
            log(
                f"Financial rows eligible for write to '{table_name}' after diff check: "
                f"{len(rows_to_write)}/{len(filtered_dataframe)}"
            )
            if rows_to_write.empty:
                log(
                    f"Skipped write for '{table_name}': no new or changed rows detected."
                )
                write_outputs[key] = rows_to_write
                continue
            self.azure_data_source.write_sql_table(
                table_name=table_name,
                engine=engine,
                df=rows_to_write,
                overwrite=False,
                dtype_overrides={
                    "DATE": satypes.Date(),
                    "REPORT_DATE": satypes.Date(),
                },
            )
            log(f"Wrote {len(rows_to_write)} rows to Azure table '{table_name}'")
            write_outputs[key] = rows_to_write

        return write_outputs


class EPSRevisionsData(YahooData):
    """Pull EPS revisions and optionally write to Azure."""

    def __init__(
        self,
        *,
        tickers: Iterable[str] | str | None = None,
        yahoo_client=None,
        client=None,
    ) -> None:
        """Initialize EPS revisions pipeline state."""
        super().__init__(tickers=tickers, yahoo_client=yahoo_client, client=client)
        self.table_name = "eps_revisions"
        self.max_yfinance_resets = 10
        self.reset_wait_seconds = 120

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
        ticker_to_figi: dict[str, str | None] | None = None,
    ) -> pd.DataFrame:
        """Run EPS revisions pull workflow with missing-ticker retries."""
        log(
            "Running EPS revisions pipeline: "
            f"{len(self.tickers)} tickers, write_to_azure={write_to_azure}"
        )
        dataframe = self._pull_with_missing_ticker_retries(
            client_method="get_eps_revisions",
            method_kwargs={},
            date_columns=["DATE"],
            max_resets=self.max_yfinance_resets,
            wait_seconds=self.reset_wait_seconds,
        )
        if "DATE" in dataframe.columns:
            dataframe = dataframe_utils.ensure_datetime_column(dataframe, "DATE")
            dataframe["DATE"] = dataframe["DATE"].dt.date
        dataframe = self._attach_figi_from_mapping(dataframe, ticker_to_figi)
        log(f"Pulled EPS revisions rows: {len(dataframe)}")

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
                f"EPS revisions rows eligible for write after diff check: "
                f"{len(rows_to_write)}/{len(dataframe)}"
            )
            if rows_to_write.empty:
                log("Skipped EPS revisions write: no new or changed rows detected.")
                return rows_to_write
            else:
                self.azure_data_source.write_sql_table(
                    table_name=self.table_name,
                    engine=engine,
                    df=rows_to_write,
                    overwrite=False,
                    dtype_overrides={"DATE": satypes.Date()},
                )
                log(f"Wrote EPS revisions rows to Azure: {len(rows_to_write)}")
                return rows_to_write

        return dataframe


class EarningsSurprisesData(YahooData):
    """Pull earnings surprises and optionally write to Azure."""

    def __init__(
        self,
        *,
        tickers: Iterable[str] | str | None = None,
        yahoo_client=None,
        client=None,
    ) -> None:
        """Initialize earnings surprises pipeline state."""
        super().__init__(tickers=tickers, yahoo_client=yahoo_client, client=client)
        self.table_name = "earnings_surprises"
        self.max_yfinance_resets = 10
        self.reset_wait_seconds = 120

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
        ticker_to_figi: dict[str, str | None] | None = None,
    ) -> pd.DataFrame:
        """Run earnings surprises pull workflow with missing-ticker retries."""
        log(
            "Running earnings surprises pipeline: "
            f"{len(self.tickers)} tickers, write_to_azure={write_to_azure}"
        )
        dataframe = self._pull_with_missing_ticker_retries(
            client_method="get_earnings_surprises",
            method_kwargs={},
            date_columns=["DATE", "EARNINGS_DATE"],
            max_resets=self.max_yfinance_resets,
            wait_seconds=self.reset_wait_seconds,
        )
        for column in ["DATE", "EARNINGS_DATE"]:
            if column in dataframe.columns:
                dataframe = dataframe_utils.ensure_datetime_column(dataframe, column)
                dataframe[column] = dataframe[column].dt.date
        dataframe = self._attach_figi_from_mapping(dataframe, ticker_to_figi)
        log(f"Pulled earnings surprises rows: {len(dataframe)}")

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
                "Earnings surprises rows eligible for write after diff check: "
                f"{len(rows_to_write)}/{len(dataframe)}"
            )
            if rows_to_write.empty:
                log(
                    "Skipped earnings surprises write: no new or changed rows detected."
                )
                return rows_to_write
            else:
                self.azure_data_source.write_sql_table(
                    table_name=self.table_name,
                    engine=engine,
                    df=rows_to_write,
                    overwrite=False,
                    dtype_overrides={
                        "DATE": satypes.Date(),
                        "EARNINGS_DATE": satypes.Date(),
                    },
                )
                log(f"Wrote earnings surprises rows to Azure: {len(rows_to_write)}")
                return rows_to_write

        return dataframe


class EstimatesData(YahooData):
    """Pull Yahoo analyst estimates (eps, revenue, growth) and optionally write to Azure."""

    _ESTIMATE_TYPES: tuple[str, ...] = ("eps", "revenue", "growth")
    _TABLE_NAME_MAP: dict[str, str] = {
        "eps": "eps_estimates",
        "revenue": "revenue_estimates",
        "growth": "growth_estimates",
    }

    def __init__(
        self,
        *,
        tickers: Iterable[str] | str | None = None,
        yahoo_client=None,
        client=None,
    ) -> None:
        """Initialize analyst estimates pipeline state."""
        super().__init__(tickers=tickers, yahoo_client=yahoo_client, client=client)
        self.max_yfinance_resets = 10
        self.reset_wait_seconds = 120

    def _pull_estimate(self, *, estimate_type: str) -> pd.DataFrame:
        """Pull one estimate type with missing-ticker retries and normalize DATE."""
        log(f"Running {estimate_type} estimates pull: " f"{len(self.tickers)} tickers")
        dataframe = self._pull_with_missing_ticker_retries(
            client_method="get_analyst_estimates",
            method_kwargs={"estimate_type": estimate_type},
            date_columns=["DATE"],
            max_resets=self.max_yfinance_resets,
            wait_seconds=self.reset_wait_seconds,
        )
        if "DATE" in dataframe.columns:
            dataframe = dataframe_utils.ensure_datetime_column(dataframe, "DATE")
            dataframe["DATE"] = dataframe["DATE"].dt.date
        log(f"Pulled {estimate_type} estimates rows: {len(dataframe)}")
        return dataframe

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
        ticker_to_figi: dict[str, str | None] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run all estimate pulls and optionally write them to Azure."""
        outputs: dict[str, pd.DataFrame] = {
            estimate_type: self._pull_estimate(estimate_type=estimate_type)
            for estimate_type in self._ESTIMATE_TYPES
        }
        outputs = {
            estimate_type: self._attach_figi_from_mapping(dataframe, ticker_to_figi)
            for estimate_type, dataframe in outputs.items()
        }

        if not write_to_azure:
            return outputs

        write_outputs: dict[str, pd.DataFrame] = {}
        engine = self.azure_data_source.get_engine(configs_path=configs_path)
        for estimate_type, dataframe in outputs.items():
            table_name = self._TABLE_NAME_MAP[estimate_type]
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
                f"{estimate_type} estimates rows eligible for write after diff check: "
                f"{len(rows_to_write)}/{len(dataframe)}"
            )
            if rows_to_write.empty:
                log(
                    f"Skipped {estimate_type} estimates write: no new or changed rows detected."
                )
                write_outputs[estimate_type] = rows_to_write
                continue

            self.azure_data_source.write_sql_table(
                table_name=table_name,
                engine=engine,
                df=rows_to_write,
                overwrite=False,
                dtype_overrides={"DATE": satypes.Date()},
            )
            log(f"Wrote {estimate_type} estimates rows to Azure: {len(rows_to_write)}")
            write_outputs[estimate_type] = rows_to_write

        return write_outputs
