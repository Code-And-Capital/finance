"""Pipeline runner for sequential datasource execution."""

from __future__ import annotations

from typing import Dict, Literal, Sequence

from data_loading.company_info_data_source import CompanyInfoDataSource
from data_loading.holdings_data_source import HoldingsDataSource
from data_loading.index_data_source import IndexDataSource
from data_loading.prices_data_source import PricesDataSource
from utils.logging import log


class Runner:
    """Run holdings, prices, and security data sources in dependency order.

    The execution order is:
    1. Holdings
    2. Prices (for holdings tickers)
    3. Security/company info (for holdings tickers)
    """

    def __init__(
        self,
        *,
        portfolio: Sequence[str] | str,
        holdings_source: Literal["index", "llm"] = "index",
        index_tickers: Sequence[str] | str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        configs_path: str | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.holdings_source = holdings_source
        self.index_tickers = index_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.configs_path = configs_path

        self.holdings_data_source: HoldingsDataSource | None = None
        self.prices_data_source: PricesDataSource | None = None
        self.index_data_source: IndexDataSource | None = None
        self.security_data_source: CompanyInfoDataSource | None = None
        self.price_dates = None

    def _run_holdings(self) -> list[str]:
        """Run holdings source and return the extracted ticker universe."""
        log("Runner: starting holdings step", type="info")
        self.holdings_data_source = HoldingsDataSource(
            source=self.holdings_source,
            portfolio=self.portfolio,
            start_date=self.start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )
        self.holdings_data_source.run()
        tickers = self.holdings_data_source.tickers
        if not tickers:
            raise ValueError("No tickers found in holdings data.")
        holdings_rows = (
            0
            if self.holdings_data_source.transformed_data is None
            else len(self.holdings_data_source.transformed_data)
        )
        log(
            f"Runner: holdings step complete rows={holdings_rows} tickers={len(tickers)}",
            type="info",
        )
        return tickers

    def _run_prices(self, tickers: Sequence[str]) -> None:
        """Run prices source for the provided holdings ticker universe."""
        log(f"Runner: starting prices step for tickers={len(tickers)}", type="info")
        self.prices_data_source = PricesDataSource(
            tickers=tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )
        self.prices_data_source.run()
        self.prices_data_source.format()
        prices_wide = self.prices_data_source.formatted_data.get("prices_wide")
        self.price_dates = prices_wide.index if prices_wide is not None else None
        if self.holdings_data_source is not None:
            self.holdings_data_source.format(dates=self.price_dates)
        prices_rows = (
            0
            if self.prices_data_source.transformed_data is None
            else len(self.prices_data_source.transformed_data)
        )
        log(f"Runner: prices step complete rows={prices_rows}", type="info")

    def _run_security(self, tickers: Sequence[str]) -> None:
        """Run security/company-info source for the provided ticker universe."""
        log(f"Runner: starting security step for tickers={len(tickers)}", type="info")
        self.security_data_source = CompanyInfoDataSource(
            tickers=tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )
        self.security_data_source.run()
        self.security_data_source.format(dates=self.price_dates)
        security_rows = (
            0
            if self.security_data_source.transformed_data is None
            else len(self.security_data_source.transformed_data)
        )
        log(f"Runner: security step complete info_rows={security_rows}", type="info")

    def _run_index(self) -> None:
        """Run optional index returns pull when index tickers are provided."""
        if self.index_tickers is None:
            log("Runner: index step skipped because index_tickers is None", type="info")
            self.index_data_source = None
            return
        log("Runner: starting index step", type="info")
        self.index_data_source = IndexDataSource(
            tickers=self.index_tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )
        self.index_data_source.run()
        self.index_data_source.format(dates=self.price_dates)
        index_rows = (
            0
            if self.index_data_source.transformed_data is None
            else len(self.index_data_source.transformed_data)
        )
        log(f"Runner: index step complete rows={index_rows}", type="info")

    def run(self) -> Dict[str, object]:
        """Run all data sources in order and return instantiated sources."""
        log(
            "Runner: run started "
            f"source={self.holdings_source} portfolio={self.portfolio} "
            f"start_date={self.start_date} end_date={self.end_date}",
            type="info",
        )
        tickers = self._run_holdings()
        self._run_prices(tickers)
        self._run_index()
        self._run_security(tickers)
        log("Runner: run completed", type="info")

        return {
            "holdings": self.holdings_data_source,
            "prices": self.prices_data_source,
            "index": self.index_data_source,
            "security": self.security_data_source,
        }
