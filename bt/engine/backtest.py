from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Callable, Dict, Optional, Any
import pyprind
from bt.core.security import SecurityBase
from bt.engine.results import Result
from ffn.core import PerformanceStats


from typing import Any, Iterable
from tqdm import tqdm


def run(*backtests: Iterable[Any], progress_bar: bool = True) -> Result:
    """
    Execute one or more backtests and return a ``Result`` object.

    Parameters
    ----------
    *backtests : Iterable
        One or more backtest objects (passed variadically). Each object must
        implement a ``run()`` method and expose the attributes expected by
        ``Result``.

    progress_bar : bool, optional
        Whether to display a progress bar while running the backtests.
        Defaults to True.

    Returns
    -------
    Result
        An aggregated result object containing each backtest's performance.
    """

    # The function signature supports both run(bt1, bt2) AND run(*[bt1, bt2]).
    # Flatten in case the user passes a single iterable.
    if len(backtests) == 1 and isinstance(backtests[0], (list, tuple)):
        backtests = tuple(backtests[0])

    # Validate inputs (optional but helpful)
    for bt_obj in backtests:
        if not hasattr(bt_obj, "run"):
            raise TypeError(
                f"Backtest object '{bt_obj}' has no 'run()' method. "
                "Each item must be a valid backtest."
            )

    # Run each backtest
    for bt_obj in tqdm(backtests, disable=not progress_bar):
        bt_obj.run()

    return Result(*backtests)


class Backtest:
    """
    Execute a strategy on historical data to produce performance results.

    A `Backtest` deep-copies the provided strategy, attaches price data and any
    supplementary datasets, then runs the strategy forward in time while
    recording values, weights, and performance statistics.

    The copied strategy ensures the original strategy instance is re-usable in
    multiple backtests.

    Parameters
    ----------
    strategy : Strategy | StrategyBase | Node
        Strategy (or strategy node) to be backtested.
    data : pd.DataFrame
        Price data used as the strategy's trading universe.
    name : str, optional
        Name of the backtest. Defaults to the strategy's name.
    initial_capital : float, default 1_000_000.0
        Initial portfolio value before the first trading day.
    commissions : Callable[[float, float], float], optional
        Commission function receiving `(quantity, price)` and returning a cost.
    integer_positions : bool, default True
        Whether the strategy trades only integer quantities.
        Set to `False` for more robust behavior with small capital or large prices.
    progress_bar : bool, default False
        Whether to display a visual progress bar during the run.
    additional_data : dict[str, Any], optional
        Additional datasets forwarded to `StrategyBase.setup`.
        These must share the same index as the input `data`.

    Attributes
    ----------
    strategy : Strategy
        Deep-copied strategy used internally by the backtest.
    data : pd.DataFrame
        Data with an additional initial NaN row for stable day-0 behavior.
    dates : pd.DatetimeIndex
        Date index of the backtest data.
    stats : ffn.PerformanceStats | dict
        Performance statistics after the run.
    weights : pd.DataFrame
        Component weights over time.
    security_weights : pd.DataFrame
        Per-security weights relative to total NAV.
    has_run : bool
        Indicates whether the backtest has already been executed.
    additional_data : dict
        Supplementary datasets passed to strategy setup.
    """

    def __init__(
        self,
        strategy: Any,
        data: pd.DataFrame,
        name: Optional[str] = None,
        initial_capital: float = 1_000_000.0,
        commissions: Optional[Callable[[float, float], float]] = None,
        integer_positions: bool = True,
        progress_bar: bool = False,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Validate duplicate columns early
        if data.columns.duplicated().any():
            dupes = data.columns[data.columns.duplicated()].tolist()
            raise ValueError(
                f"Data contains duplicate columns:\n{dupes}\nPlease ensure uniqueness."
            )

        # Deep copy strategy so the original can be reused
        self.strategy = deepcopy(strategy)
        self.strategy.use_integer_positions(integer_positions)

        # Preprocess prices + additional_data
        self._process_data(data, additional_data)

        self.initial_capital = initial_capital
        self.name = name or strategy.name
        self.progress_bar = progress_bar

        if commissions is not None:
            self.strategy.set_commissions(commissions)

        # Internal placeholders
        self.stats = {}
        self._original_prices = None
        self._weights = None
        self._sweights = None
        self.has_run = False

    def _process_data(
        self,
        data: pd.DataFrame,
        additional_data: Optional[Dict[str, Any]],
    ) -> None:
        """
        Prepare data for the backtest by inserting a NaN placeholder row at t₀−1.
        This ensures initial trades (day 0) are calculated relative to a clean
        baseline, preventing day-0 return distortion.

        Additional datasets with the same index as `data` are also prepended
        with a matching NaN row.
        """
        # Insert NaN row at t0 - 1 day
        t0 = data.index[0] - pd.DateOffset(days=1)
        prepend = pd.DataFrame(np.nan, columns=data.columns, index=[t0])
        data_new = pd.concat([prepend, data])

        self.data = data_new
        self.dates = data_new.index

        # Normalize additional data
        self.additional_data = (additional_data or {}).copy()

        for k in self.additional_data:
            old = self.additional_data[k]
            if isinstance(old, pd.DataFrame) and old.index.equals(data.index):
                empty_row = pd.DataFrame(
                    np.nan,
                    columns=old.columns,
                    index=[old.index[0] - pd.DateOffset(days=1)],
                )
                # Ensure dtypes match to avoid FutureWarning
                empty_row = empty_row.astype(old.dtypes)
                new = pd.concat([empty_row, old])
                self.additional_data[k] = new
            elif isinstance(old, pd.Series) and old.index.equals(data.index):
                empty_row = pd.Series(
                    np.nan,
                    index=[old.index[0] - pd.DateOffset(days=1)],
                    dtype=old.dtype,
                )
                new = pd.concat([empty_row, old])
                self.additional_data[k] = new

    def run(self) -> None:
        """
        Execute the backtest.

        The strategy is initialized, capital is applied, and each timestamp is
        processed sequentially. After execution, performance statistics and
        price series are stored internally.
        """
        if self.has_run:
            return

        self.has_run = True

        # Strategy initialization
        self.strategy.setup(self.data, **self.additional_data)
        self.strategy.adjust(self.initial_capital)

        # Optional progress bar
        bar = (
            pyprind.ProgBar(len(self.dates), title=self.name, stream=1)
            if self.progress_bar
            else None
        )

        # Day 0 update (dummy row)
        self.strategy.update(self.dates[0])

        # Main loop
        for dt in self.dates[1:]:
            if bar:
                bar.update()

            self.strategy.update(dt)

            if not self.strategy.bankrupt:
                self.strategy.run()
                self.strategy.update(dt)
            else:
                if bar:
                    bar.stop()
                break

        self.stats = PerformanceStats(self.strategy.prices)
        self._original_prices = self.strategy.prices

    # ----------------------------------------------------------------------

    @property
    def weights(self) -> pd.DataFrame:
        """
        Component weights over time.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each column is a portfolio component's weight.
        """
        if self._weights is not None:
            return self._weights

        if self.strategy.fixed_income:
            vals = pd.DataFrame(
                {m.full_name: m.notional_values for m in self.strategy.members}
            )
            vals = vals.div(self.strategy.notional_values, axis=0)
        else:
            vals = pd.DataFrame({m.full_name: m.values for m in self.strategy.members})
            vals = vals.div(self.strategy.values, axis=0)

        self._weights = vals
        return vals

    @property
    def positions(self) -> pd.DataFrame:
        """
        Raw security positions over time.

        Returns
        -------
        pd.DataFrame
            Position quantities for each component.
        """
        return self.strategy.positions

    # ----------------------------------------------------------------------

    @property
    def security_weights(self) -> pd.DataFrame:
        """
        Per-security weights as a percentage of total NAV.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each column is a security and each row contains
            weights summing to 1.
        """
        if self._sweights is not None:
            return self._sweights

        vals = {}

        for member in self.strategy.members:
            if isinstance(member, SecurityBase):
                series = (
                    member.notional_values.copy()
                    if self.strategy.fixed_income
                    else member.values.copy()
                )
                if member.name in vals:
                    vals[member.name] += series
                else:
                    vals[member.name] = series
        df = pd.DataFrame(vals)

        if self.strategy.fixed_income:
            df = df.div(self.strategy.notional_values, axis=0)
        else:
            df = df.div(self.strategy.values, axis=0)

        self._sweights = df
        return df

    @property
    def herfindahl_index(self) -> pd.Series:
        """
        Herfindahl-Hirschman Index (HHI) of portfolio concentration.

        HHI = Σ(wᵢ²).
        Low HHI = diversified, High HHI = concentrated.

        Returns
        -------
        pd.Series
            Daily HHI values.
        """
        w = self.security_weights
        return (w**2).sum(axis=1)

    # ----------------------------------------------------------------------

    @property
    def turnover(self) -> pd.Series:
        """
        Portfolio turnover.

        Turnover is defined as:

            min( Σ positive outflows, Σ |negative outflows| ) / NAV

        Returns
        -------
        pd.Series
            Daily turnover values.
        """
        s = self.strategy
        outlays = s.outlays

        pos = outlays[outlays >= 0].fillna(0).sum(axis=1)
        neg = outlays[outlays < 0].fillna(0).abs().sum(axis=1)

        min_outlay = pd.DataFrame({"pos": pos, "neg": neg}).min(axis=1)
        return min_outlay / s.values
