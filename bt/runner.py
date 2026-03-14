"""Runner utilities for backtests and data-loading pipelines."""

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import pandas as pd
from tqdm import tqdm

from bt.analytics import BacktestSummary
from bt.core.backtest import Backtest
from bt.core.commission import zero_commission
from bt.core.strategy import Strategy
from utils.logging import log
from visualization.charts import Line
from visualization.figure import Figure

if TYPE_CHECKING:
    from data_loading.company_info_data_source import CompanyInfoDataSource
    from data_loading.holdings_data_source import HoldingsDataSource
    from data_loading.index_data_source import IndexDataSource
    from data_loading.prices_data_source import PricesDataSource


class Runner:
    """Run holdings, prices, and security data sources in dependency order."""

    def __init__(
        self,
        *,
        portfolio: Sequence[str] | str,
        holdings_source: Literal["index", "llm"] = "index",
        index_figis: Sequence[str] | str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        configs_path: str | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.holdings_source = holdings_source
        self.index_figis = index_figis
        self.start_date = start_date
        self.end_date = end_date
        self.configs_path = configs_path

        self.holdings_data_source: HoldingsDataSource | None = None
        self.prices_data_source: PricesDataSource | None = None
        self.index_data_source: IndexDataSource | None = None
        self.security_data_source: CompanyInfoDataSource | None = None
        self.price_dates = None
        self.figi_to_ticker: dict[str, str] = {}
        self.ticker_to_figi: dict[str, str] = {}

    def _load_holdings(self) -> list[str]:
        """Run holdings source and return the extracted FIGI universe."""
        from data_loading.holdings_data_source import HoldingsDataSource

        log("Runner: starting holdings step", type="info")
        self.holdings_data_source = HoldingsDataSource(
            source=self.holdings_source,
            portfolio=self.portfolio,
            start_date=self.start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )
        self.holdings_data_source.load()
        figis = self.holdings_data_source.figis
        if not figis:
            raise ValueError("No FIGIs found in holdings data.")
        holdings_rows = (
            0
            if self.holdings_data_source.transformed_data is None
            else len(self.holdings_data_source.transformed_data)
        )
        log(
            f"Runner: holdings step complete rows={holdings_rows} figis={len(figis)}",
            type="info",
        )
        return figis

    def _load_prices(self, figis: Sequence[str]) -> None:
        """Run prices source for the provided holdings FIGI universe."""
        from data_loading.prices_data_source import PricesDataSource

        log(f"Runner: starting prices step for figis={len(figis)}", type="info")
        self.prices_data_source = PricesDataSource(
            figis=figis,
            start_date=self.start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )
        self.prices_data_source.load()
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

    def _load_security(self, figis: Sequence[str]) -> None:
        """Run security/company-info source for the provided FIGI universe."""
        from data_loading.company_info_data_source import CompanyInfoDataSource

        log(f"Runner: starting security step for figis={len(figis)}", type="info")
        self.security_data_source = CompanyInfoDataSource(
            figis=figis,
            start_date=self.start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )
        self.security_data_source.load()
        self.security_data_source.format(dates=self.price_dates)
        security_rows = (
            0
            if self.security_data_source.transformed_data is None
            else len(self.security_data_source.transformed_data)
        )
        log(f"Runner: security step complete info_rows={security_rows}", type="info")

    def _load_index(self) -> None:
        """Run optional index returns pull when index FIGIs are provided."""
        from data_loading.index_data_source import IndexDataSource

        if self.index_figis is None:
            log("Runner: index step skipped because index_figis is None", type="info")
            self.index_data_source = None
            return
        log("Runner: starting index step", type="info")
        self.index_data_source = IndexDataSource(
            figis=self.index_figis,
            start_date=self.start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )
        self.index_data_source.load()
        self.index_data_source.format(dates=self.price_dates)
        index_rows = (
            0
            if self.index_data_source.transformed_data is None
            else len(self.index_data_source.transformed_data)
        )
        log(f"Runner: index step complete rows={index_rows}", type="info")

    def load_data(self) -> dict[str, object]:
        """Run all data sources in order and return instantiated sources."""
        log(
            "Runner: run started "
            f"source={self.holdings_source} portfolio={self.portfolio} "
            f"start_date={self.start_date} end_date={self.end_date}",
            type="info",
        )
        figis = self._load_holdings()
        self._load_prices(figis)
        self._load_index()
        self._load_security(figis)
        self._build_symbol_mappings()
        log("Runner: run completed", type="info")

        return {
            "holdings": self.holdings_data_source,
            "prices": self.prices_data_source,
            "index": self.index_data_source,
            "security": self.security_data_source,
        }

    def _build_symbol_mappings(self) -> None:
        """Build FIGI/ticker lookup maps from formatted company info data."""
        self.figi_to_ticker = {}
        self.ticker_to_figi = {}

        if self.security_data_source is not None:
            ticker_wide = self.security_data_source.formatted_data.get("ticker_wide")
            if isinstance(ticker_wide, pd.DataFrame) and not ticker_wide.empty:
                for figi in ticker_wide.columns:
                    series = ticker_wide[figi].dropna()
                    if series.empty:
                        continue
                    ticker = str(series.iloc[-1]).strip().upper()
                    normalized_figi = str(figi).strip().upper()
                    if not ticker:
                        continue
                    self.figi_to_ticker[normalized_figi] = ticker
                    self.ticker_to_figi.setdefault(ticker, normalized_figi)

        if self.index_data_source is not None:
            index_prices_long = self.index_data_source.formatted_data.get(
                "index_prices_long"
            )
            if (
                isinstance(index_prices_long, pd.DataFrame)
                and not index_prices_long.empty
                and {"FIGI", "TICKER"}.issubset(index_prices_long.columns)
            ):
                mapping_rows = index_prices_long.loc[:, ["FIGI", "TICKER"]].dropna()
                for row in mapping_rows.itertuples(index=False):
                    normalized_figi = str(row.FIGI).strip().upper()
                    ticker = str(row.TICKER).strip().upper()
                    if not normalized_figi or not ticker:
                        continue
                    self.figi_to_ticker.setdefault(normalized_figi, ticker)
                    self.ticker_to_figi.setdefault(ticker, normalized_figi)

    def _resolve_plot_figis(
        self,
        *,
        figis: Sequence[str] | None,
        tickers: Sequence[str] | None,
        available_figis: pd.Index,
    ) -> list[str]:
        """Resolve plot identifiers into an ordered FIGI list."""
        if figis is not None and tickers is not None:
            raise ValueError("Provide only one of `figis` or `tickers`.")

        available = {str(figi).strip().upper() for figi in available_figis}
        requested: list[str] = []

        if figis is not None:
            requested = [
                str(figi).strip().upper() for figi in figis if str(figi).strip()
            ]
            missing = [figi for figi in requested if figi not in available]
            if missing:
                log(
                    "Runner: requested plot FIGIs not found: " + ", ".join(missing),
                    type="warning",
                )
            requested = [figi for figi in requested if figi in available]
        elif tickers is not None:
            normalized_tickers = [
                str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()
            ]
            missing = [
                ticker
                for ticker in normalized_tickers
                if ticker not in self.ticker_to_figi
            ]
            if missing:
                log(
                    "Runner: requested plot tickers not found: " + ", ".join(missing),
                    type="warning",
                )
            for ticker in normalized_tickers:
                figi = self.ticker_to_figi.get(ticker)
                if figi is not None and figi in available:
                    requested.append(figi)
        else:
            requested = list(available_figis)

        if not requested:
            raise ValueError(
                "None of the requested securities are present in prices_wide."
            )

        return requested

    def _display_name_for_figi(self, figi: str) -> str:
        """Return a human-readable series label for a FIGI."""
        ticker = self.figi_to_ticker.get(figi)
        if ticker is None:
            return figi
        if self.ticker_to_figi.get(ticker) == figi:
            return ticker
        return f"{ticker} ({figi})"

    def _rename_benchmark_columns(self, benchmark: pd.DataFrame) -> pd.DataFrame:
        """Rename benchmark columns from FIGI identifiers to ticker labels."""
        renamed = benchmark.copy()
        renamed.columns = [
            self.figi_to_ticker.get(str(column).strip().upper(), str(column))
            for column in benchmark.columns
        ]
        return renamed

    def plot_prices(
        self,
        *,
        figis: Sequence[str] | None = None,
        tickers: Sequence[str] | None = None,
        use_full_history: bool = False,
        title: str = "Prices",
        height: int = 500,
    ) -> Figure:
        """Create a line chart for rebased prices by FIGI or ticker."""
        if (
            self.prices_data_source is None
            or not self.prices_data_source.formatted_data
        ):
            raise ValueError("load_data must be called before plot_prices.")

        key = "prices_wide_full_history" if use_full_history else "prices_wide"
        prices = self.prices_data_source.formatted_data.get(key)
        if not isinstance(prices, pd.DataFrame):
            raise ValueError(f"{key} is unavailable; cannot plot prices.")
        if prices.empty:
            raise ValueError(f"{key} is empty; nothing to plot.")

        selected_figis = self._resolve_plot_figis(
            figis=figis,
            tickers=tickers,
            available_figis=prices.columns,
        )
        plot_data = prices.loc[:, selected_figis].copy()
        plot_data.columns = [
            self._display_name_for_figi(figi) for figi in selected_figis
        ]

        line = Line(plot_data)
        line.create(x="index", y=list(plot_data.columns), width=2, mode="lines")
        line.quick_styling(
            x_title="Date",
            y_title="Price",
            selector_buttons=False,
            rangeslider=False,
        )

        fig = Figure(rows=1, cols=1)
        fig.add_chart(line, row=1, col=1)
        fig.layout(title=title, height=height, showlegend=True)
        fig.show()
        return fig

    def _collect_additional_data(self) -> dict[str, Any]:
        """Collect formatted datasource outputs for strategy setup."""
        additional_data: dict[str, Any] = {}
        for source in (
            self.holdings_data_source,
            self.prices_data_source,
            self.index_data_source,
            self.security_data_source,
        ):
            if source is not None:
                additional_data.update(source.formatted_data)
        return additional_data

    @staticmethod
    def run_strategies(
        *backtests: Backtest,
        benchmark: pd.DataFrame | None = None,
        figi_to_ticker: dict[str, str] | None = None,
        progress_bar: bool = True,
    ) -> BacktestSummary:
        """Run the provided backtests and return a combined summary."""
        if not backtests:
            raise ValueError("run_strategies requires at least one backtest.")

        for backtest in tqdm(backtests, disable=not progress_bar):
            backtest.run()

        return BacktestSummary(
            *backtests,
            benchmark=benchmark,
            figi_to_ticker=figi_to_ticker,
        )

    def run_backtest(
        self,
        strategies: Strategy | Sequence[Strategy],
        integer_positions: bool = False,
        progress_bar: bool = True,
        commissions: Callable[[float, float], float] = zero_commission,
    ) -> BacktestSummary:
        """
        Build backtests for the provided strategies and execute them.

        Parameters
        ----------
        strategies
            Strategies to backtest against the runner's prepared price data.
        integer_positions
            Whether backtests should restrict positions to integer quantities.
        progress_bar
            Whether to display the outer backtest progress bar.
        commissions
            Commission function passed through to each constructed backtest.
        """
        if isinstance(strategies, Strategy):
            strategies = [strategies]
        else:
            strategies = list(strategies)

        if not strategies:
            raise ValueError("run requires at least one strategy.")

        if (
            self.prices_data_source is None
            or not self.prices_data_source.formatted_data
        ):
            raise ValueError("load_data must be called before run_backtest.")

        prices = self.prices_data_source.formatted_data.get("prices_wide_full_history")
        if not isinstance(prices, pd.DataFrame) or prices.empty:
            raise ValueError(
                "Runner prices_wide_full_history is empty; cannot build backtests."
            )
        live_prices = self.prices_data_source.formatted_data.get("prices_wide")
        if not isinstance(live_prices, pd.DataFrame) or live_prices.empty:
            raise ValueError(
                "Runner prices_wide is empty; cannot determine live trading start."
            )
        live_start_date = live_prices.index[0]

        additional_data = self._collect_additional_data()
        benchmark = None
        if self.index_data_source is not None:
            index_prices = self.index_data_source.formatted_data.get(
                "index_prices_wide"
            )
            if isinstance(index_prices, pd.DataFrame) and not index_prices.empty:
                benchmark = self._rename_benchmark_columns(index_prices)
        backtests = [
            Backtest(
                strategy=strategy,
                prices=prices,
                commissions=commissions,
                integer_positions=integer_positions,
                progress_bar=False,
                additional_data=additional_data,
                live_start_date=live_start_date,
            )
            for strategy in strategies
        ]
        return self.run_strategies(
            *backtests,
            benchmark=benchmark,
            figi_to_ticker=self.figi_to_ticker,
            progress_bar=progress_bar,
        )
