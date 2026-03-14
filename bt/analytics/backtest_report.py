"""Backtest summary container built on analytics primitives."""

from typing import Any, List, Optional, Union

import pandas as pd

from .group import MultiSeriesPerformanceStats
from visualization.charts import Area
from visualization.charts import Bar
from visualization.charts import Line
from visualization.charts import Pie
from visualization.figure import Figure


class BacktestSummary(MultiSeriesPerformanceStats):
    """
    Container for executed backtests with analytics and retrieval helpers.

    This class extends :class:`MultiSeriesPerformanceStats` by attaching
    backtest objects and exposing convenience accessors for common strategy
    artifacts (weights, positions, transactions, etc.).

    Parameters
    ----------
    backtests : Any
        One or more executed backtest objects. Each backtest is expected to
        expose ``name`` and ``strategy.prices``.
    benchmark : pandas.DataFrame, optional
        Benchmark price frame stored on the summary for downstream reporting.
    figi_to_ticker : dict[str, str], optional
        Optional mapping used to relabel FIGI-based outputs for display.
    rf : float | pandas.Series, optional
        Risk-free input passed to ``MultiSeriesPerformanceStats``.
    annualization_factor : int, optional
        Annualization factor passed to ``MultiSeriesPerformanceStats``.
    """

    def __init__(
        self,
        *backtests: Any,
        benchmark: pd.DataFrame | None = None,
        figi_to_ticker: dict[str, str] | None = None,
        rf: float | pd.Series = 0.0,
        annualization_factor: int = 252,
    ) -> None:
        if not backtests:
            raise ValueError("BacktestSummary requires at least one backtest.")

        names: list[str] = []
        price_series: list[pd.Series] = []
        for bt in backtests:
            name = getattr(bt, "name", None)
            strategy = getattr(bt, "strategy", None)
            prices = getattr(strategy, "prices", None) if strategy is not None else None

            if not isinstance(name, str) or not name:
                raise TypeError("Each backtest must expose a non-empty string `name`.")
            if not isinstance(prices, pd.Series):
                raise TypeError(
                    f"Backtest '{name}' must expose `strategy.prices` as a pandas Series."
                )
            if prices.empty:
                raise ValueError(
                    f"Backtest '{name}' has an empty `strategy.prices` series."
                )
            if not isinstance(prices.index, pd.DatetimeIndex):
                raise TypeError(
                    f"Backtest '{name}' `strategy.prices` index must be a DatetimeIndex."
                )
            names.append(name)
            price_series.append(prices.rename(name))

        if len(set(names)) != len(names):
            raise ValueError("Backtest names must be unique in BacktestSummary.")
        if benchmark is not None:
            if not isinstance(benchmark, pd.DataFrame):
                raise TypeError("benchmark must be a pandas DataFrame.")
            if benchmark.empty:
                raise ValueError("benchmark must be a non-empty DataFrame.")
            if not isinstance(benchmark.index, pd.DatetimeIndex):
                raise TypeError("benchmark index must be a pandas DatetimeIndex.")

        self.backtest_list = list(backtests)
        self.backtests = {bt.name: bt for bt in backtests}
        self.benchmark = benchmark.copy() if benchmark is not None else None
        self.figi_to_ticker = (figi_to_ticker or {}).copy()

        performance_inputs: list[pd.Series | pd.DataFrame] = list(price_series)
        if self.benchmark is not None:
            performance_inputs.append(self.benchmark)

        super().__init__(
            *performance_inputs,
            rf=rf,
            annualization_factor=annualization_factor,
        )

    def _get_backtest(self, backtest: Union[str, int]) -> str:
        """
        Resolve a backtest selector to a backtest name.

        Parameters
        ----------
        backtest : str | int
            Backtest selector by name or positional index.

        Returns
        -------
        str
            Resolved backtest name.
        """
        if isinstance(backtest, int):
            if backtest < 0 or backtest >= len(self.backtest_list):
                raise IndexError(
                    f"Backtest index {backtest} out of range "
                    f"(0 to {len(self.backtest_list)-1})."
                )
            return self.backtest_list[backtest].name

        if backtest not in self.backtests:
            raise KeyError(f"Backtest name '{backtest}' not found.")
        return backtest

    def get_weights(
        self,
        backtest: Union[str, int] = 0,
        filter: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Return component weights for a selected backtest.

        Parameters
        ----------
        backtest : str | int, optional
            Backtest selector by name or positional index.
        filter : str | list[str], optional
            Optional column selector applied to the returned DataFrame.

        Returns
        -------
        pandas.DataFrame
            Component-level weights over time.
        """
        key = self._get_backtest(backtest)
        data = self.backtests[key].weights
        return data[filter] if filter is not None else data

    def get_security_weights(
        self,
        backtest: Union[str, int] = 0,
        filter: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Return per-security weights for a selected backtest.

        Parameters
        ----------
        backtest : str | int, optional
            Backtest selector by name or positional index.
        filter : str | list[str], optional
            Optional column selector applied to the returned DataFrame.

        Returns
        -------
        pandas.DataFrame
            Security-level weights over time.
        """
        key = self._get_backtest(backtest)
        data = self.backtests[key].security_weights
        return data[filter] if filter is not None else data

    def get_data(self, backtest: Union[str, int] = 0) -> pd.DataFrame:
        """
        Return strategy data for a selected backtest.

        Parameters
        ----------
        backtest : str | int, optional
            Backtest selector by name or positional index.

        Returns
        -------
        pandas.DataFrame
            Strategy data frame (e.g. price/value/cash/flow fields) stored on
            ``backtest.strategy.data``.
        """
        key = self._get_backtest(backtest)
        data = self.backtests[key].strategy.data
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Backtest '{key}' strategy data must be a pandas DataFrame."
            )
        return data

    def get_universe(self, backtest: Union[str, int] = 0) -> pd.DataFrame:
        """
        Return underlying universe prices for a selected backtest.

        Parameters
        ----------
        backtest : str | int, optional
            Backtest selector by name or positional index.

        Returns
        -------
        pandas.DataFrame
            Universe price DataFrame stored on ``backtest.strategy.universe``.
        """
        key = self._get_backtest(backtest)
        data = self.backtests[key].strategy.universe
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Backtest '{key}' strategy universe must be a pandas DataFrame."
            )
        return data

    def get_positions(self, backtest: Union[str, int] = 0) -> pd.DataFrame:
        """
        Return security positions for a selected backtest.

        Parameters
        ----------
        backtest : str | int, optional
            Backtest selector by name or positional index.

        Returns
        -------
        pandas.DataFrame
            Positions DataFrame stored on ``backtest.positions``.
        """
        key = self._get_backtest(backtest)
        data = self.backtests[key].positions
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Backtest '{key}' positions must be a pandas DataFrame.")
        return data

    def get_outlays(self, backtest: Union[str, int] = 0) -> pd.DataFrame:
        """
        Return strategy outlays for a selected backtest.

        Outlays are the dollar amounts spent (buy) or received (sell) from
        security transactions. Rows with all-zero outlays are removed.

        Parameters
        ----------
        backtest : str | int, optional
            Backtest selector by name or positional index.

        Returns
        -------
        pandas.DataFrame
            Outlays DataFrame with all-zero rows filtered out.
        """
        key = self._get_backtest(backtest)
        data = self.backtests[key].strategy.outlays
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Backtest '{key}' strategy outlays must be a pandas DataFrame."
            )
        mask = (data != 0).any(axis=1)
        return data.loc[mask]

    def get_transactions(
        self, strategy_name: Optional[Union[str, int]] = None
    ) -> pd.DataFrame:
        """
        Return transaction records for a selected backtest.

        Parameters
        ----------
        strategy_name : str | int | None, optional
            Backtest selector by name or positional index. If ``None``, the
            first backtest in ``backtest_list`` is used.

        Returns
        -------
        pandas.DataFrame
            Transaction log with a MultiIndex
            ``(Date, Security) -> quantity, price``.
        """
        if strategy_name is None:
            key = self.backtest_list[0].name
        else:
            key = self._get_backtest(strategy_name)
        return self.backtests[key].get_transactions

    def plot_prices(
        self,
        *,
        height: int = 500,
    ) -> Figure:
        """Plot all strategy and benchmark price series in one figure."""
        all_prices = self.prices

        line = Line(all_prices)
        line.create(x="index", y=list(all_prices.columns), width=2, mode="lines")
        line.quick_styling(
            x_title="Date",
            y_title="Price",
            selector_buttons=False,
            rangeslider=False,
        )
        line.legend(show=True)

        fig = Figure(rows=1, cols=1)
        fig.add_chart(line, row=1, col=1)
        fig.layout(title="Cumulative Returns", height=height, showlegend=True)
        fig.show()
        return fig

    def plot_security_weights(
        self,
        backtest: Union[str, int] = 0,
        *,
        height: int = 500,
    ) -> Figure:
        """Plot stacked security weights for a selected backtest."""
        key = self._get_backtest(backtest)
        df_weights = self.get_security_weights(key)
        display_weights = df_weights.copy()
        display_weights.columns = [
            self.figi_to_ticker.get(str(column).strip().upper(), str(column))
            for column in df_weights.columns
        ]

        area = Area(display_weights)
        area.create(x="index", y=list(display_weights.columns), stacked=True)
        area.quick_styling(
            x_title="Date",
            y_title="Price",
            selector_buttons=False,
            rangeslider=False,
        )
        area.yaxis(tickformat=".0%")

        fig = Figure(rows=1, cols=1)
        fig.add_chart(area, row=1, col=1)
        fig.layout(title=f"Security Weights - {key}", height=height, showlegend=True)
        fig.show()
        return fig

    def plot_current_security_weights(
        self,
        backtest: Union[str, int] = 0,
        *,
        height: int = 500,
        hole: float = 0.3,
    ) -> Figure:
        """Plot the latest security weights for a selected backtest as a pie chart."""
        key = self._get_backtest(backtest)
        current_weights = self.get_security_weights(key).iloc[-1]
        current_weights = current_weights[current_weights != 0].sort_values(
            ascending=False
        )
        display_weights = current_weights.rename(
            index=lambda value: self.figi_to_ticker.get(
                str(value).strip().upper(), str(value)
            )
        )

        pie = Pie(display_weights.to_frame(name="WEIGHT"))
        pie.create(x="index", y="WEIGHT", hole=hole, textinfo="label")

        fig = Figure(rows=1, cols=1)
        fig.add_chart(pie, row=1, col=1)
        fig.layout(
            title=f"Current Security Weights - {key}",
            height=height,
            showlegend=True,
        )
        fig.show()
        return fig

    def plot_sector_weights(
        self,
        sector_wide: pd.DataFrame,
        backtest: Union[str, int] = 0,
        *,
        height: int = 500,
    ) -> Figure:
        """Aggregate security weights into sector weights and plot them."""
        key = self._get_backtest(backtest)
        if not isinstance(sector_wide, pd.DataFrame):
            raise TypeError("sector_wide must be a pandas DataFrame.")
        if sector_wide.empty:
            raise ValueError("sector_wide is empty; nothing to plot.")

        weights_wide = self.get_security_weights(key)
        common_idx = weights_wide.index.intersection(sector_wide.index)
        common_cols = weights_wide.columns.intersection(sector_wide.columns)
        if common_idx.empty or common_cols.empty:
            raise ValueError(
                "sector_wide does not overlap with security weights on dates/columns."
            )

        weights_aligned = weights_wide.loc[common_idx, common_cols]
        sectors_aligned = sector_wide.loc[common_idx, common_cols]

        weights_long = weights_aligned.stack().rename("WEIGHT").reset_index()
        sectors_long = sectors_aligned.stack().rename("SECTOR").reset_index()
        weights_long.columns = ["DATE", "FIGI", "WEIGHT"]
        sectors_long.columns = ["DATE", "FIGI", "SECTOR"]

        sector_weights = (
            weights_long.merge(sectors_long, on=["DATE", "FIGI"], how="inner")
            .dropna(subset=["WEIGHT", "SECTOR"])
            .groupby(["DATE", "SECTOR"], as_index=False)["WEIGHT"]
            .sum()
            .pivot(index="DATE", columns="SECTOR", values="WEIGHT")
            .sort_index()
            .fillna(0.0)
        )
        if sector_weights.empty:
            raise ValueError("No sector weights available after alignment.")

        area = Area(sector_weights)
        area.create(x="index", y=list(sector_weights.columns), stacked=True)
        area.quick_styling(
            x_title="Date",
            y_title="Weight",
            selector_buttons=False,
            rangeslider=False,
        )
        area.yaxis(tickformat=".0%")

        fig = Figure(rows=1, cols=1)
        fig.add_chart(area, row=1, col=1)
        fig.layout(title=f"Sector Weights - {key}", height=height, showlegend=True)
        fig.show()
        return fig

    def plot_factor_stats(
        self,
        factor: str,
        backtest: Union[str, int] = 0,
        *,
        height: int = 500,
    ) -> Figure:
        """Plot MEAN, MEDIAN, 25TH, and 75TH factor stats for a selected backtest."""
        key = self._get_backtest(backtest)
        algo = self.backtests[key].strategy.algos.get(factor)
        if algo is None:
            raise KeyError(f"Factor '{factor}' not found in backtest '{key}'.")

        stats = getattr(algo, "stats", None)
        if not isinstance(stats, pd.DataFrame):
            raise TypeError(f"Factor '{factor}' stats must be a pandas DataFrame.")
        if stats.empty:
            raise ValueError(f"Factor '{factor}' stats are empty; nothing to plot.")

        plot_stats = stats.loc[:, ["MEAN", "MEDIAN", "25TH", "75TH"]].copy()

        line = Line(plot_stats)
        line.create(x="index", y=list(plot_stats.columns), width=2, mode="lines")
        for trace in line.traces:
            if trace.name in {"25TH", "75TH"}:
                trace.line = trace.line or {}
                trace.line["dash"] = "dot"
        line.quick_styling(
            x_title="Date",
            y_title="Value",
            selector_buttons=False,
            rangeslider=False,
        )
        line.legend(show=True)

        fig = Figure(rows=1, cols=1)
        fig.add_chart(line, row=1, col=1)
        fig.layout(
            title=f"Factor Stats - {factor} - {key}", height=height, showlegend=True
        )
        fig.show()
        return fig

    def plot_signal_state_counts(
        self,
        signal: str,
        backtest: Union[str, int] = 0,
        tickers: list[str] | None = None,
        *,
        height: int = 500,
    ) -> Figure:
        """Plot selected/not-selected counts for a signal over time."""
        key = self._get_backtest(backtest)
        algo = self.backtests[key].strategy.algos.get(signal)
        if algo is None:
            raise KeyError(f"Signal '{signal}' not found in backtest '{key}'.")

        history = getattr(algo, "history", None)
        if not isinstance(history, pd.DataFrame):
            raise TypeError(f"Signal '{signal}' history must be a pandas DataFrame.")
        if history.empty:
            raise ValueError(f"Signal '{signal}' history is empty; nothing to plot.")

        display_history = history.copy()
        display_history.columns = [
            self.figi_to_ticker.get(str(column).strip().upper(), str(column))
            for column in history.columns
        ]

        if tickers is not None:
            requested = [
                str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()
            ]
            missing = [
                ticker for ticker in requested if ticker not in display_history.columns
            ]
            if missing:
                raise KeyError(
                    f"Requested tickers not found in signal history for '{signal}': {missing}"
                )
            display_history = display_history.loc[:, requested]

        counts = display_history.apply(lambda series: series.value_counts()).T.fillna(0)
        counts.columns = [str(column) for column in counts.columns]

        bar = Bar(counts)
        bar.create(x="index", y=list(counts.columns))

        fig = Figure(rows=1, cols=1)
        fig.add_chart(bar, row=1, col=1)
        fig.layout(
            title=f"Signal Counts - {signal} - {key}", height=height, showlegend=True
        )
        fig.show()
        return fig

    def plot_signal_selection_counts(
        self,
        signal: str,
        backtest: Union[str, int] = 0,
        top_n: int = 50,
        *,
        height: int = 500,
    ) -> Figure:
        """Plot top signal selection counts across the full history."""
        if top_n <= 0:
            raise ValueError("top_n must be greater than 0.")

        key = self._get_backtest(backtest)
        algo = self.backtests[key].strategy.algos.get(signal)
        if algo is None:
            raise KeyError(f"Signal '{signal}' not found in backtest '{key}'.")

        history = getattr(algo, "history", None)
        if not isinstance(history, pd.DataFrame):
            raise TypeError(f"Signal '{signal}' history must be a pandas DataFrame.")
        if history.empty:
            raise ValueError(f"Signal '{signal}' history is empty; nothing to plot.")

        persistence = history.sum(axis=0).sort_values(ascending=False).iloc[:top_n]
        persistence.name = "Count"
        persistence.index = persistence.index.map(
            lambda figi: self.figi_to_ticker.get(str(figi).strip().upper(), figi)
        )

        bar_data = pd.DataFrame(persistence)
        bar = Bar(bar_data)
        bar.create(x="index", y="Count")
        bar.quick_styling(
            x_title="Ticker",
            y_title="Count",
            selector_buttons=False,
            rangeslider=False,
        )

        fig = Figure(rows=1, cols=1)
        fig.add_chart(bar, row=1, col=1)
        fig.layout(
            title=f"Signal Selection Counts - {signal} - {key}",
            height=height,
            showlegend=False,
        )
        fig.show()
        return fig
