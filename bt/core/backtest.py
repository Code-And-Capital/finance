"""Backtest execution primitives for strategies and price data."""

from copy import deepcopy
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm

from bt.core.commission import zero_commission
from bt.core.strategy import Strategy


class Backtest:
    """
    Execute a strategy against historical data.

    ``Backtest`` deep-copies the provided strategy, attaches price data and any
    supplementary datasets, then runs the strategy forward in time while
    recording values, weights, and summary performance statistics. The copied
    strategy keeps the caller's original strategy reusable across backtests.

    Parameters
    ----------
    strategy : Strategy
        Strategy object to be backtested.
    prices : pd.DataFrame
        Price data used as the strategy's trading universe. Must be a non-empty
        DataFrame indexed by a monotonic increasing ``DatetimeIndex``.
    name : str, optional
        Name of the backtest. Defaults to the strategy's name.
    initial_capital : float, default 1_000_000.0
        Initial portfolio value before the first trading day. Must be finite.
    commissions : Callable[[float, float], float], default zero_commission
        Commission function receiving `(quantity, price)` and returning a cost.
    integer_positions : bool, default False
        Whether the strategy trades only integer quantities.
        Set to `True` when whole-share execution is required.
    progress_bar : bool, default False
        Whether to display a progress bar for this backtest's date loop.
    additional_data : Mapping[str, Any], optional
        Additional datasets forwarded to `Strategy.setup` unchanged.
    live_start_date : str | pandas.Timestamp, optional
        First date of the aligned live trading window. When provided, the
        backtest seeds state at this date's close and only starts executing
        strategy logic from the following date. This allows full-history
        prices to be used as warm-up data without requiring explicit
        scheduling algos in the strategy stack.

    Attributes
    ----------
    strategy : Strategy
        Deep-copied strategy used internally by the backtest.
    data : pd.DataFrame
        Price data passed to the strategy during setup.
    dates : pd.DatetimeIndex
        Date index of the backtest data.
    has_run : bool
        Indicates whether the backtest has completed successfully.
    additional_data : dict[str, Any]
        Supplementary datasets passed to strategy setup.
    """

    def __init__(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        name: str | None = None,
        initial_capital: float = 1_000_000.0,
        commissions: Callable[[float, float], float] = zero_commission,
        integer_positions: bool = False,
        progress_bar: bool = False,
        additional_data: Mapping[str, Any] | None = None,
        live_start_date: str | pd.Timestamp | None = None,
    ) -> None:
        self._validate_prices(prices)
        self._process_data(prices, additional_data)
        self._validate_initial_capital(initial_capital)

        self.strategy = deepcopy(strategy)
        self.strategy.use_integer_positions(integer_positions)

        self.initial_capital = initial_capital
        self.name = self._resolve_backtest_name(name)
        self.progress_bar = progress_bar
        self._live_start_index = self._resolve_live_start_index(live_start_date)
        self.live_start_date = self.dates[self._live_start_index]

        self.strategy.set_commissions(commissions)

        self._weights = None
        self._sweights = None
        self.has_run = False

    @staticmethod
    def _validate_prices(prices: pd.DataFrame) -> None:
        """Validate the primary input price frame."""
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame.")
        if prices.empty:
            raise ValueError("prices must be a non-empty DataFrame.")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices index must be a pandas DatetimeIndex.")
        if not prices.index.is_unique:
            raise ValueError("prices index must be unique.")
        if not prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be sorted in increasing order.")
        if prices.columns.duplicated().any():
            dupes = prices.columns[prices.columns.duplicated()].tolist()
            raise ValueError(
                f"Data contains duplicate columns:\n{dupes}\nPlease ensure uniqueness."
            )

    def _process_data(
        self,
        prices: pd.DataFrame,
        additional_data: Mapping[str, Any] | None,
    ) -> None:
        """
        Store the primary prices and supplementary inputs.
        """
        self.data = prices
        self.dates = prices.index
        self.additional_data = (additional_data or {}).copy()

    @staticmethod
    def _validate_initial_capital(initial_capital: float) -> None:
        """Validate the initial capital value."""
        if not np.isfinite(initial_capital):
            raise ValueError("initial_capital must be finite.")

    def _resolve_backtest_name(self, explicit_name: str | None) -> str:
        """Resolve a stable backtest name."""
        if explicit_name:
            return explicit_name

        candidate = getattr(self.strategy, "name", None)
        if isinstance(candidate, str) and candidate:
            return candidate

        return "backtest"

    def _resolve_live_start_index(
        self,
        live_start_date: str | pd.Timestamp | None,
    ) -> int:
        """Resolve the close date used to seed the live trading window."""
        if live_start_date is None:
            return 0

        try:
            live_start_ts = pd.Timestamp(live_start_date)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "live_start_date must be parseable as a timestamp."
            ) from exc

        if live_start_ts not in self.dates:
            raise ValueError("live_start_date must be present in the prices index.")

        return int(self.dates.get_loc(live_start_ts))

    def run(self) -> None:
        """
        Execute the backtest.

        The strategy is initialized, capital is applied, and state is first
        seeded to the configured live-window start close. Strategy execution
        then starts on the following date, so pre-market trades always size
        against a prior close while still allowing earlier full-history data to
        exist purely as warm-up input. If the root strategy becomes bankrupt
        after a close, the backtest stops with that close as the terminal
        recorded state.
        """
        if self.has_run:
            return

        self.strategy.setup(
            self.data,
            live_start_date=self.live_start_date,
            **self.additional_data,
        )
        self.strategy.adjust(self.initial_capital)

        seed_index = self._live_start_index
        seed_date = self.dates[seed_index]
        self.strategy.pre_market_update(seed_date, seed_index)
        self.strategy.post_market_update()

        date_iter = tqdm(
            enumerate(self.dates[seed_index + 1 :], start=seed_index + 1),
            disable=not self.progress_bar,
            desc=self.name,
            leave=False,
            total=max(len(self.dates) - (seed_index + 1), 0),
        )
        for inow, dt in date_iter:
            self.strategy.pre_market_update(dt, inow)
            self.strategy.run()
            self.strategy.post_market_update()

            if self.strategy.bankrupt:
                break

        self.has_run = True

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

        vals = pd.DataFrame({m.full_name: m.values for m in self.strategy.members})
        vals = vals.div(self.strategy.values, axis=0)

        self._weights = vals
        return vals

    @property
    def security_weights(self) -> pd.DataFrame:
        """
        Each security's weight over time.

        Returns
        -------
        pd.DataFrame
            Weight of securities.
        """
        if self._sweights is not None:
            return self._sweights

        vals = {}
        for member in self.strategy.members:
            if member._issec:
                member_values = member.values.copy()
                if member.name in vals:
                    vals[member.name] += member_values
                else:
                    vals[member.name] = member_values
            else:
                if "Cash" in vals:
                    vals["Cash"] += member.cash
                else:
                    vals["Cash"] = member.cash

        vals = pd.DataFrame(vals)
        vals = vals.div(self.strategy.values, axis=0)

        self._sweights = vals
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

    @property
    def get_transactions(self) -> pd.DataFrame:
        """Return transactions as a MultiIndex DataFrame."""
        securities = self.strategy.securities
        if not securities:
            return pd.DataFrame(columns=["price", "quantity"])

        prices = pd.DataFrame(
            {security.name: security.prices for security in securities}
        ).unstack()
        positions = self.positions

        if positions.empty:
            return pd.DataFrame(columns=["price", "quantity"])

        trades = positions.diff()
        trades.iloc[0] = positions.iloc[0]
        trades = trades[trades != 0].unstack().dropna()

        if trades.empty:
            return pd.DataFrame(columns=["price", "quantity"])

        result = pd.DataFrame({"price": prices, "quantity": trades}).dropna(
            subset=["quantity"]
        )
        result.index.names = ["Security", "Date"]
        return result.swaplevel().sort_index()
