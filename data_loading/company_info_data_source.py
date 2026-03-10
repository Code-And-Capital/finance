"""Datasource for company metadata."""

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd

from data_loading.base_data_source import BaseDataSource
from handyman.company_info import get_company_info
from utils.logging import log


class CompanyInfoDataSource(BaseDataSource):
    """Load and format company info datasets."""

    def __init__(
        self,
        *,
        figis: Sequence[str] | str,
        start_date: str | None = None,
        end_date: str | None = None,
        configs_path: str | None = None,
    ) -> None:
        super().__init__()
        self.figis = figis
        self.start_date = start_date
        self.end_date = end_date
        self.configs_path = configs_path

    def load(self) -> pd.DataFrame:
        """Load company info data."""
        log(
            "CompanyInfoDataSource: loading info for "
            f"figis_count={len(self.figis) if isinstance(self.figis, (list, tuple, set)) else 1} "
            f"start_date={self.start_date} end_date={self.end_date}",
            type="info",
        )
        info_df = get_company_info(
            figis=self.figis,
            start_date=self.start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )
        log(
            f"CompanyInfoDataSource: loaded info_rows={len(info_df)}",
            type="info",
        )
        return info_df

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize FIGI/date fields and deduplicate rows."""
        log(f"CompanyInfoDataSource: transforming info rows={len(data)}", type="info")
        requested = self._requested_figis()
        out = data.copy()
        if "DATE" in out.columns:
            out["DATE"] = pd.to_datetime(out["DATE"])
        if "FIGI" in out.columns:
            out["FIGI"] = out["FIGI"].astype(str).str.strip().str.upper()
        out = out.drop_duplicates().reset_index(drop=True)
        if requested:
            found = set(
                out.get("FIGI", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .unique()
            )
            missing = sorted(set(requested) - found)
            if missing:
                log(
                    "CompanyInfoDataSource: missing data for requested FIGIs: "
                    + ", ".join(missing),
                    type="warning",
                )
        log(f"CompanyInfoDataSource: transformed info rows={len(out)}", type="info")
        return out

    def format(
        self,
        dates: Sequence[pd.Timestamp] | pd.Index | None = None,
    ) -> None:
        """Populate ``self.formatted_data`` with company info payload."""
        if self.transformed_data is None:
            raise ValueError("run() must be called before format().")
        log(
            f"CompanyInfoDataSource: formatting info rows={len(self.transformed_data)}",
            type="info",
        )
        outputs: Dict[str, pd.DataFrame] = {
            "company_info": self.transformed_data.copy()
        }
        data = self.transformed_data
        if {"DATE", "FIGI", "SECTOR"}.issubset(data.columns):
            sector_wide = data.pivot_table(
                index="DATE",
                columns="FIGI",
                values="SECTOR",
                aggfunc="last",
            ).sort_index()
            if dates is not None:
                target_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
                sector_wide = sector_wide.reindex(target_dates)
            sector_wide = sector_wide.ffill().bfill()
            outputs["sector_wide"] = sector_wide
        if {"DATE", "FIGI", "TICKER"}.issubset(data.columns):
            ticker_wide = data.pivot_table(
                index="DATE",
                columns="FIGI",
                values="TICKER",
                aggfunc="last",
            ).sort_index()
            if dates is not None:
                target_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
                ticker_wide = ticker_wide.reindex(target_dates)
            ticker_wide = ticker_wide.ffill().bfill()
            outputs["ticker_wide"] = ticker_wide
        if {"DATE", "FIGI", "MARKETCAP"}.issubset(data.columns):
            marketcap_wide = data.pivot_table(
                index="DATE",
                columns="FIGI",
                values="MARKETCAP",
                aggfunc="last",
            ).sort_index()
            if dates is not None:
                target_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
                marketcap_wide = marketcap_wide.reindex(target_dates)
            marketcap_wide = marketcap_wide.ffill().bfill()
            outputs["marketcap_wide"] = marketcap_wide
        self.formatted_data = outputs
        log(
            f"CompanyInfoDataSource: formatted company_info rows={len(outputs['company_info'])}",
            type="info",
        )
