from __future__ import annotations

import pandas as pd
from typing import Optional, Union, List, Tuple, Any, Iterable, Dict
import matplotlib.pyplot as plt

import ffn


class Result(ffn.GroupStats):
    """
    Container for one or more strategy backtests with convenience methods for
    inspecting weights, returns, and transactions.

    This class extends ``ffn.GroupStats`` and adds a number of helper utilities
    for visualizing and extracting values from individual backtests.

    Parameters
    ----------
    backtests : Backtest
        One or more ``Backtest`` objects.

    Attributes
    ----------
    backtest_list : list[Backtest]
        Backtests in the same order they were provided.
    backtests : dict[str, Backtest]
        Mapping of backtest name → backtest instance.
    """

    def __init__(self, *backtests: Any) -> None:
        # ffn.GroupStats expects multiple DataFrames of price series
        price_frames = [pd.DataFrame({bt.name: bt.strategy.prices}) for bt in backtests]

        super().__init__(*price_frames)

        self.backtest_list = list(backtests)
        self.backtests = {bt.name: bt for bt in backtests}

    def display_monthly_returns(self, backtest: Union[str, int] = 0) -> None:
        """
        Display monthly returns for a given backtest.

        Parameters
        ----------
        backtest : str or int, default 0
            Name of the backtest or its index within ``backtest_list``.
        """
        key = self._get_backtest(backtest)
        self[key].display_monthly_returns()

    def get_weights(
        self,
        backtest: Union[str, int] = 0,
        filter: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Return component weights for a given backtest.

        Parameters
        ----------
        backtest : str or int, default 0
            Name or index of the backtest.
        filter : str or list[str], optional
            Column selection passed directly to DataFrame indexing.

        Returns
        -------
        pd.DataFrame
            Component weights over time.
        """
        key = self._get_backtest(backtest)

        data = self.backtests[key].weights
        return data[filter] if filter is not None else data

    # ----------------------------------------------------------------------

    def plot_weights(
        self,
        backtest: Union[str, int] = 0,
        filter: Optional[Union[str, List[str]]] = None,
        figsize: tuple[int, int] = (15, 5),
        **kwds: Any,
    ) -> None:
        """
        Plot component weights of the selected backtest.

        Parameters
        ----------
        backtest : str or int, default 0
            Name or index of the backtest.
        filter : str or list[str], optional
            Columns to plot.
        figsize : tuple(int, int), default (15, 5)
            Figure dimensions.
        kwds : dict
            Additional keyword arguments forwarded to ``DataFrame.plot``.
        """
        self.get_weights(backtest, filter).plot(figsize=figsize, **kwds)

    def get_security_weights(
        self,
        backtest: Union[str, int] = 0,
        filter: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Return per-security weights of the selected backtest.

        Parameters
        ----------
        backtest : str or int, default 0
            Name or index of the backtest.
        filter : str or list[str], optional
            Columns to select.

        Returns
        -------
        pd.DataFrame
            Security weights over time.
        """
        key = self._get_backtest(backtest)
        data = self.backtests[key].security_weights
        return data[filter] if filter is not None else data

    def plot_security_weights(
        self,
        backtest: Union[str, int] = 0,
        filter: Optional[Union[str, List[str]]] = None,
        figsize: tuple[int, int] = (15, 5),
        **kwds: Any,
    ) -> None:
        """
        Plot per-security weights over time.

        Parameters
        ----------
        backtest : str or int, default 0
            Name or index of the backtest.
        filter : str or list[str], optional
            Columns to select.
        figsize : tuple(int, int), default (15, 5)
            Figure size.
        kwds : dict
            Extra keyword args passed to ``DataFrame.plot``.
        """
        self.get_security_weights(backtest, filter).plot(figsize=figsize, **kwds)

    def plot_histogram(self, backtest: Union[str, int] = 0, **kwds: Any) -> None:
        """
        Plot a histogram of returns for the selected backtest.

        Parameters
        ----------
        backtest : str or int, default 0
            Name or index of the backtest.
        kwds : dict
            Extra arguments passed to ``ffn.PerformanceStats.plot_histogram``.
        """
        key = self._get_backtest(backtest)
        self[key].plot_histogram(**kwds)

    def _get_backtest(self, backtest: Union[str, int]) -> str:
        """
        Normalize a backtest selector (name or index) to its name.

        Parameters
        ----------
        backtest : str or int
            Backtest name or position in ``backtest_list``.

        Returns
        -------
        str
            The name of the selected backtest.
        """
        if isinstance(backtest, int):
            if backtest >= len(self.backtest_list):
                raise IndexError(
                    f"Backtest index {backtest} out of range "
                    f"(0 to {len(self.backtest_list)-1})."
                )
            return self.backtest_list[backtest].name

        # Assume valid name
        if backtest not in self.backtests:
            raise KeyError(f"Backtest name '{backtest}' not found.")
        return backtest

    def get_transactions(self, strategy_name: Optional[str] = None) -> pd.DataFrame:
        """
        Return the full transactions log for a given strategy.

        The returned DataFrame has a MultiIndex of:

            (Date, Security) → quantity, price

        Parameters
        ----------
        strategy_name : str, optional
            Backtest name.
            If omitted, the first backtest in ``backtest_list`` is used.

        Returns
        -------
        pd.DataFrame
            MultiIndex transaction log from ``strategy.get_transactions()``.
        """
        if strategy_name is None:
            strategy_name = self.backtest_list[0].name

        if strategy_name not in self.backtests:
            raise KeyError(f"Strategy '{strategy_name}' not found in Result.")

        return self.backtests[strategy_name].strategy.get_transactions()


class RandomBenchmarkResult(Result):
    """
    Extends `Result` to provide utilities for evaluating a strategy
    against a series of randomly generated benchmark strategies.

    Args:
        *backtests: Variable number of backtest objects. The first is
            considered the benchmarked (base) strategy, and the rest
            are random strategies.

    Attributes:
        base_name (str):
            Name of the benchmarked backtest.
        r_stats (pd.DataFrame):
            Statistics for the random strategies only.
        b_stats (pd.Series):
            Statistics for the benchmarked strategy.
    """

    def __init__(self, *backtests: Any) -> None:
        if not backtests:
            raise ValueError("At least one backtest must be provided.")

        super().__init__(*backtests)

        self.base_name: str = backtests[0].name

        # Separate benchmark stats and random stats
        if self.base_name not in self.stats.columns:
            raise KeyError(
                f"Benchmark name '{self.base_name}' not found in stats columns: "
                f"{list(self.stats.columns)}"
            )

        self.b_stats = self.stats[self.base_name]
        self.r_stats = self.stats.drop(self.base_name, axis=1)

    def plot_histogram(
        self,
        statistic: str = "monthly_sharpe",
        figsize: Tuple[int, int] = (15, 5),
        title: Optional[str] = None,
        bins: int = 20,
        **kwargs,
    ) -> None:
        """
        Plot a histogram showing the distribution of a given statistic across
        the random strategies. A red vertical line shows the benchmark strategy's
        value.

        This helps determine whether the benchmark strategy is statistically
        superior compared to the distribution of random alternatives.

        Args:
            statistic:
                The statistic to plot. Must exist in the stats index.
            figsize:
                Matplotlib figure size.
            title:
                Chart title. If None, a default title is generated.
            bins:
                Number of histogram bins.
            **kwargs:
                Additional keyword arguments passed to pandas' `hist()`.

        Raises:
            ValueError: If the statistic does not exist.
        """

        if statistic not in self.r_stats.index:
            valid = list(self.r_stats.index)
            raise ValueError(
                f"Invalid statistic '{statistic}'. Valid statistics: {valid}"
            )

        ser = self.r_stats.loc[statistic]

        if title is None:
            title = f"{statistic} distribution"

        plt.figure(figsize=figsize)

        # Histogram of random-strategy distribution
        ax = ser.hist(bins=bins, density=True, **kwargs)
        ax.set_title(title)

        # Benchmark line
        benchmark_value = self.b_stats.get(statistic)
        if pd.isna(benchmark_value):
            raise ValueError(
                f"Statistic '{statistic}' not found in benchmark stats for '{self.base_name}'."
            )

        plt.axvline(
            benchmark_value,
            linewidth=4,
            color="r",
            label=f"Benchmark ({self.base_name})",
        )

        # KDE curve
        ser.plot(kind="kde")

        plt.legend()
        plt.show()


class RenormalizedFixedIncomeResult(Result):
    """
    A result type used to recompute normalized price series for
    fixed-income strategies whose notionals vary over time.

    Background
    ----------
    Fixed-income strategies in `bt` compute normalized prices using
    additive returns scaled by the *current* outstanding notional.
    When notionals vary, this can lead to unintuitive normalized price
    behavior (e.g., price < PAR while actual value increases).

    This class allows you to recompute the price series using a fixed
    or alternative normalizing denominator (e.g., constant notional,
    average notional, max exposure, or any custom series).

    Parameters
    ----------
    normalizing_value : Union[pd.Series, float, Dict[str, Union[pd.Series, float]]]
        A value or timeseries used to renormalize the price calculations.
        If not a dict, the same value is applied to all backtests.
        If a dict, must map backtest names → normalizing value.

    *backtests : Iterable
        A list of backtest objects originating from `Result.backtest_list`,
        each of which must be based on a fixed-income strategy.

    Attributes
    ----------
    backtest_list : list
        List of supplied backtest objects.

    backtests : dict
        Mapping of {backtest_name → backtest object} for convenience.
    """

    def __init__(
        self,
        normalizing_value: Union[pd.Series, float, Dict[str, Union[pd.Series, float]]],
        *backtests: Iterable[Any],
    ) -> None:
        # --- Validate backtests -------------------------------------------------
        for bt_obj in backtests:
            if not bt_obj.strategy.fixed_income:
                raise ValueError(
                    f"Cannot apply RenormalizedFixedIncomeResult because "
                    f"backtest '{bt_obj.name}' does not use a fixed-income strategy."
                )

        # --- Normalize the normalizing_value input -----------------------------
        if not isinstance(normalizing_value, dict):
            # broadcast same value to all
            normalizing_value = {bt_obj.name: normalizing_value for bt_obj in backtests}

        # Ensure all backtests have an entry
        missing = [
            bt_obj.name for bt_obj in backtests if bt_obj.name not in normalizing_value
        ]
        if missing:
            raise KeyError(
                f"Missing normalizing_value entries for: {missing}. "
                f"Provide a dict mapping strategy names to values."
            )

        # --- Build the DataFrames for each backtest ----------------------------
        tmp_frames = []
        for bt_obj in backtests:
            name = bt_obj.name
            strategy = bt_obj.strategy
            normalizer = normalizing_value[name]

            price_series = self._price(strategy, normalizer)
            tmp_frames.append(pd.DataFrame({name: price_series}))

        # Initialize base Result class
        super(Result, self).__init__(*tmp_frames)

        # Store references
        self.backtest_list = list(backtests)
        self.backtests = {bt_obj.name: bt_obj for bt_obj in backtests}

    # ----------------------------------------------------------------------
    def _price(self, strategy: Any, normalizer: Union[pd.Series, float]) -> pd.Series:
        """
        Compute the renormalized price series for a fixed-income strategy.

        Parameters
        ----------
        strategy : Strategy
            Fixed-income strategy whose values and flows are used.

        normalizer : Union[pd.Series, float]
            Denominator used to compute additive returns.

        Returns
        -------
        pd.Series
            Renormalized price series beginning at PAR.
        """

        # additive return net of flows
        additive_returns = strategy.values.diff() - strategy.flows

        # cumulative normalized returns
        cumulative = (additive_returns / normalizer).cumsum()

        prices = 100 * (1.0 + cumulative)
        prices.iloc[0] = 100

        return prices
