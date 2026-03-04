"""Pipeline component for downloading and preparing index holdings."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

import utils.dataframe_utils as dataframe_utils
from utils.logging import log
from connectors.azure_data_source import default_azure_data_source
from connectors.selenium_data_source import default_selenium_data_source
from connectors.xls_data_source import default_xls_data_source
from sql.script_factory import SQLClient

ETF_URLS = {
    "S&P 500": "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf",
    "Russell 1000": "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf",
    # "Russell 3000": "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf",
    # "MSCI World": "https://www.ishares.com/us/products/239696/ishares-msci-world-etf",
}

ETF_FILE_NAMES = {
    "S&P 500": "iShares-Core-SP-500-ETF_fund.xls",
    "Russell 1000": "iShares-Russell-1000-ETF_fund.xls",
    "Russell 3000": "iShares-Russell-3000-ETF_fund.xls",
}

EXPECTED_ROW_BOUNDS = {
    "S&P 500": (450, 550),
    "Russell 1000": (900, 1100),
}

DEFAULT_DOWNLOAD_FOLDER = (Path(__file__).resolve().parents[2] / "Data").resolve()

TICKER_MAPPING = {
    "LENB": "LEN-B",
    "GEFB": "GEF-B",
    "BRKB": "BRK-B",
    "HEIA": "HEI-A",
    "BFB": "BF-B",
    "UHALB": "UHAL-B",
    "MOGA": "MOG-A",
    "BFA": "BF-A",
    "CWENA": "CWEN-A",
    "GLIBR": "GLIBK",
    "VSNTV UW": "VSNT",
    "VSNT*": "VSNT",
}


class DownloadHoldings:
    """Download and transform index holdings data into the target schema."""

    def __init__(
        self,
        *,
        fund_name: str,
        url: str,
        download_folder: str | None = None,
    ) -> None:
        """Initialize the holdings downloader pipeline component."""
        self.fund_name = fund_name
        self.url = url
        resolved_folder = (
            Path(download_folder).expanduser()
            if download_folder
            else DEFAULT_DOWNLOAD_FOLDER
        )
        self.download_folder = str(resolved_folder.resolve())
        Path(self.download_folder).mkdir(parents=True, exist_ok=True)
        self.azure_data_source = default_azure_data_source
        self.sql_client = SQLClient()
        self.selenium_data_source = default_selenium_data_source
        self.xls_data_source = default_xls_data_source
        self.table_name = "holdings"
        self.allowed_exchanges = [
            "NASDAQ",
            "New York Stock Exchange Inc.",
            "Nyse Mkt Llc",
            "NYSE",
        ]

    def _resolve_file_path(self) -> str:
        """Resolve downloaded holdings file path from fund mapping."""
        try:
            file_name = ETF_FILE_NAMES[self.fund_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown fund_name '{self.fund_name}' in ETF_FILE_NAMES"
            ) from exc
        return os.path.join(self.download_folder, file_name)

    def _download(self) -> str:
        """Trigger the Selenium download flow and return expected file path."""
        log(f"Downloading holdings for {self.fund_name}")
        self.selenium_data_source.download_data_file(
            url=self.url,
            download_folder=self.download_folder,
        )
        return self._resolve_file_path()

    def _load_raw(self, file_path: str) -> pd.DataFrame:
        """Read the downloaded XLS file into a raw DataFrame."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Downloaded holdings file not found at: {file_path}"
            )
        raw_df = self.xls_data_source.read_xls_file(
            file_path=file_path,
            sheet_number=1,
            skiprows=7,
        )
        log(f"Loaded raw holdings for {self.fund_name}: {len(raw_df)} rows")
        return raw_df

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filtering, normalization, and aggregation logic."""
        out = df[df["ASSET_CLASS"] == "Equity"]
        out = out[out["TICKER"] != "--"]
        out["TICKER"] = out["TICKER"].replace(TICKER_MAPPING)

        out = out[out["EXCHANGE"].isin(self.allowed_exchanges)]

        out = dataframe_utils.convert_columns_to_numeric(out)
        out = out[out["WEIGHT"] > 0]
        out["WEIGHT"] = out["WEIGHT"] / out["WEIGHT"].sum()

        out["INDEX"] = self.fund_name
        out["DATE"] = date.today()

        out = out[
            [
                "DATE",
                "INDEX",
                "TICKER",
                "NAME",
                "MARKET_VALUE",
                "WEIGHT",
                "QUANTITY",
                "PRICE",
                "LOCATION",
                "EXCHANGE",
                "CURRENCY",
                "FX_RATE",
            ]
        ]

        return (
            out.groupby(["DATE", "INDEX", "TICKER"])
            .agg(
                {
                    "NAME": "first",
                    "MARKET_VALUE": "sum",
                    "WEIGHT": "sum",
                    "QUANTITY": "sum",
                    "PRICE": "sum",
                    "LOCATION": "first",
                    "EXCHANGE": "first",
                    "CURRENCY": "first",
                    "FX_RATE": "first",
                }
            )
            .reset_index()
        )

    def _validate_row_count(self, df: pd.DataFrame) -> None:
        """Validate transformed row count for known index funds."""
        bounds = EXPECTED_ROW_BOUNDS.get(self.fund_name)
        if bounds is None:
            return

        min_rows, max_rows = bounds
        row_count = len(df)
        if not (min_rows <= row_count <= max_rows):
            raise ValueError(
                f"{self.fund_name} row-count check failed: got {row_count}, "
                f"expected between {min_rows} and {max_rows}"
            )

    def run(
        self,
        *,
        write_to_azure: bool = False,
        configs_path: str | None = None,
    ) -> pd.DataFrame:
        """Execute the full holdings download and transformation workflow."""
        file_path = self._download()
        try:
            raw = self._load_raw(file_path)
            transformed = self._transform(raw)
            log(f"Transformed holdings for {self.fund_name}: {len(transformed)} rows")
            self._validate_row_count(transformed)
            log(
                f"Row-count validation passed for {self.fund_name}: {len(transformed)} rows"
            )
            if write_to_azure:
                engine = self.azure_data_source.get_engine(configs_path=configs_path)
                self.azure_data_source.write_sql_table(
                    engine=engine,
                    table_name=self.table_name,
                    df=transformed,
                    overwrite=False,
                )
                log(
                    f"Wrote {len(transformed)} holdings rows to Azure for {self.fund_name}"
                )
            return transformed
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                log(
                    f"Removed downloaded holdings file for {self.fund_name}: {file_path}"
                )

    def delete_rows_by_where_clause(
        self,
        *,
        where_clause: str,
        configs_path: str | None = None,
    ) -> None:
        """Delete holdings rows matching a WHERE clause from Azure SQL."""
        log(f"Deleting holdings rows with where clause: {where_clause}")
        engine = self.azure_data_source.get_engine(configs_path=configs_path)
        delete_query = self.sql_client.build_delete_query(
            table_name=self.table_name,
            where_clause=where_clause,
            schema="dbo",
        )
        self.azure_data_source.delete_sql_rows(query=delete_query, engine=engine)
        log("Delete holdings query executed successfully")
