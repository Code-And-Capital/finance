"""Datasource for index holdings data."""

from __future__ import annotations

from typing import Dict, Literal, Sequence

import pandas as pd

from data_loading.base_data_source import BaseDataSource
from handyman.holdings import get_index_holdings, get_llm_holdings
from utils.logging import log


class HoldingsDataSource(BaseDataSource):
    """Load, transform, and format holdings from index or LLM strategy sources."""

    def __init__(
        self,
        *,
        source: Literal["index", "llm"] = "index",
        portfolio: Sequence[str] | str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        configs_path: str | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.portfolio = portfolio
        self.start_date = start_date
        self.end_date = end_date
        self.configs_path = configs_path
        self.figis: list[str] = []

        if self.source not in {"index", "llm"}:
            raise ValueError("source must be either 'index' or 'llm'.")
        if self.portfolio is None:
            raise ValueError("portfolio must be provided.")

    def load(self) -> pd.DataFrame:
        """Load holdings in long format from selected source."""
        log(
            "HoldingsDataSource: loading "
            f"source={self.source} portfolio={self.portfolio} "
            f"start_date={self.start_date} end_date={self.end_date}",
            type="info",
        )
        if self.source == "index":
            return get_index_holdings(
                indices=self.portfolio,
                start_date=self.start_date,
                end_date=self.end_date,
                configs_path=self.configs_path,
            )
        return get_llm_holdings(
            llms=self.portfolio,
            start_date=self.start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize fields, enforce ordering, and store distinct holdings FIGIs."""
        log(f"HoldingsDataSource: transforming {len(data)} rows", type="info")
        if data.empty:
            self.figis = []
            log("HoldingsDataSource: received empty holdings dataframe", type="warning")
            return data.copy()
        out = data.copy()
        if "DATE" in out.columns:
            out["DATE"] = pd.to_datetime(out["DATE"])
        if "FIGI" not in out.columns:
            raise ValueError("Holdings data must include a FIGI column.")

        figi_series = out["FIGI"]
        normalized_figis = (
            figi_series[figi_series.notna()].astype(str).str.strip().str.upper()
        )
        normalized_figis = normalized_figis[~normalized_figis.isin(["", "NAN", "NONE"])]
        out["FIGI"] = normalized_figis.reindex(out.index)
        self.figis = normalized_figis.drop_duplicates().tolist()
        if not self.figis:
            raise ValueError("No FIGIs found in holdings data.")
        log(
            f"HoldingsDataSource: extracted {len(self.figis)} unique FIGIs",
            type="info",
        )

        order_cols = [col for col in ["DATE", "FIGI"] if col in out.columns]
        if order_cols:
            out = out.sort_values(order_cols).reset_index(drop=True)
        return out

    @staticmethod
    def _ffill_single_all_nan_gaps(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill only isolated one-row gaps where all columns are missing."""
        if df.empty or len(df.index) < 3:
            return df

        out = df.copy()
        all_missing = out.isna().all(axis=1)
        for i in range(1, len(out.index) - 1):
            if (
                all_missing.iloc[i]
                and (not all_missing.iloc[i - 1])
                and (not all_missing.iloc[i + 1])
            ):
                out.iloc[i] = out.iloc[i - 1]
        return out

    def format(self, dates: Sequence[pd.Timestamp] | pd.Index | None = None) -> None:
        """Populate ``self.formatted_data`` with long + wide holdings representations."""
        if self.transformed_data is None:
            raise ValueError("run() must be called before format().")
        data = self.transformed_data
        log(
            f"HoldingsDataSource: formatting {len(data)} transformed rows",
            type="info",
        )
        outputs: Dict[str, pd.DataFrame] = {"holdings_long": data.copy()}
        if {"DATE", "FIGI", "WEIGHT"}.issubset(data.columns):
            weights_wide = data.pivot_table(
                index="DATE",
                columns="FIGI",
                values="WEIGHT",
                aggfunc="last",
            )
            in_portfolio_wide = weights_wide > 0
            if dates is not None:
                target_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
                weights_wide = weights_wide.reindex(target_dates)
                weights_wide = self._ffill_single_all_nan_gaps(weights_wide)
                in_portfolio_wide = (
                    (weights_wide > 0).reindex(target_dates).fillna(False)
                )
            outputs["weights_wide"] = weights_wide
            outputs["in_portfolio_wide"] = in_portfolio_wide
            log(
                "HoldingsDataSource: generated weights_wide "
                f"shape={weights_wide.shape} and in_portfolio_wide "
                f"shape={in_portfolio_wide.shape}",
                type="info",
            )
        self.formatted_data = outputs
