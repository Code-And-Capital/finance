"""Datasource for index-level prices and returns."""

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd

from data_loading.base_data_source import BaseDataSource
from data_loading.prices_data_source import PricesDataSource
from utils.logging import log


class IndexDataSource(BaseDataSource):
    """Load index prices via ``PricesDataSource`` and derive return series."""

    def __init__(
        self,
        *,
        tickers: Sequence[str] | str,
        start_date: str | None = None,
        end_date: str | None = None,
        configs_path: str | None = None,
    ) -> None:
        super().__init__()
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.configs_path = configs_path
        self.prices_data_source: PricesDataSource | None = None

    def load(self) -> pd.DataFrame:
        """Load index prices through the shared prices datasource."""
        log(
            "IndexDataSource: loading prices "
            f"start_date={self.start_date} end_date={self.end_date}",
            type="info",
        )
        self.prices_data_source = PricesDataSource(
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )
        prices_long = self.prices_data_source.run()
        self.prices_data_source.format()
        return prices_long

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute index return series from adjusted close prices."""
        log(f"IndexDataSource: transforming {len(data)} rows", type="info")
        if data.empty:
            log("IndexDataSource: received empty dataframe", type="warning")
            out = data.copy()
            out["RETURN"] = pd.Series(dtype="float64")
            return out

        required = {"DATE", "TICKER", "ADJ_CLOSE"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(
                f"Expected columns {sorted(required)}; missing {sorted(missing)}"
            )

        out = data.copy()
        out["DATE"] = pd.to_datetime(out["DATE"])
        out["TICKER"] = out["TICKER"].astype(str).str.upper()
        out = out.sort_values(["TICKER", "DATE"]).reset_index(drop=True)
        out["RETURN"] = out.groupby("TICKER", sort=False)["ADJ_CLOSE"].pct_change()
        return out

    def format(self, dates: Sequence[pd.Timestamp] | pd.Index | None = None) -> None:
        """Populate formatted index price and return outputs."""
        if self.transformed_data is None:
            raise ValueError("run() must be called before format().")
        data = self.transformed_data
        outputs: Dict[str, pd.DataFrame] = {"index_returns_long": data.copy()}

        if self.prices_data_source is not None:
            outputs["index_prices_long"] = self.prices_data_source.formatted_data.get(
                "prices_long", pd.DataFrame()
            )
            index_prices_wide = self.prices_data_source.formatted_data.get(
                "prices_wide", pd.DataFrame()
            )
            if dates is not None and not index_prices_wide.empty:
                target_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
                index_prices_wide = index_prices_wide.reindex(target_dates)
            outputs["index_prices_wide"] = index_prices_wide

        if data.empty:
            self.formatted_data = outputs
            return

        self.formatted_data = outputs
        log("IndexDataSource: generated index return outputs", type="info")
