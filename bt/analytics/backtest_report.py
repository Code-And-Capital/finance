"""Backtest summary container built on analytics primitives."""

from typing import Any, List, Optional, Union

import pandas as pd

from .group import MultiSeriesPerformanceStats
from visualization.charts import Area
from visualization.charts import Line
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
