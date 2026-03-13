from typing import Any

import numpy as np
import pandas as pd

from bt.algos.core import AlgoStack
from bt.core.commission import zero_commission
from bt.core.nodes import Node
from bt.core.security import Security
from utils.math_utils import is_zero


class Strategy(Node):
    """
    Strategy node with optional algo execution.

    A strategy owns capital, maintains a tradable universe, allocates capital
    to child strategies or securities, and can execute an ``AlgoStack`` during
    the pre-market trading phase.
    """

    _is_strategy = True

    @staticmethod
    def _build_algo_list(algos: list[Any]) -> list[Any]:
        """Prepend standard lifecycle algos and avoid duplicate auto-close rules."""
        from bt.algos.flow.close_positions import ClosePositionsAfterDates

        has_auto_close = any(
            isinstance(algo, ClosePositionsAfterDates)
            and getattr(algo, "_close_name", None) == "last_valid_date"
            for algo in algos
        )
        if has_auto_close:
            return algos
        return [ClosePositionsAfterDates("last_valid_date"), *algos]

    def __init__(
        self,
        name: str,
        algos: list[Any] | None = None,
        children: list[Any] | dict[str, Any] | None = None,
        PAR: float = 100.0,
    ) -> None:
        super().__init__(name=name, children=children)
        if algos is None:
            algos = []
        algos = self._build_algo_list(algos)

        self._weight = 1.0
        self._price = float(PAR)

        self._net_flows = 0.0
        self._last_value = 0.0
        self._last_price = float(PAR)
        self._last_fee = 0.0
        self._positions = None
        self.bankrupt = False
        self._bankrupt_at = None

        self.commission_fn = zero_commission
        self._setup_kwargs: dict[str, Any] = {}
        self.stack = AlgoStack(*algos)
        self.algos = {algo.name: algo for algo in algos}

        self.temp: dict[str, Any] = {}
        self.perm: dict[str, Any] = {"closed": set()}

    def _history_to_now(
        self, data: pd.Series | pd.DataFrame
    ) -> pd.Series | pd.DataFrame:
        """Return a history slice through the current timestamp."""
        if self.now == 0:
            return data.iloc[0:0]
        return data.loc[: self.now]

    @property
    def price(self) -> float:
        """Return the current strategy price."""
        return self._price

    @property
    def prices(self) -> pd.Series:
        """Return strategy price history up to ``now``."""
        return self._history_to_now(self.data["price"])

    @property
    def values(self) -> pd.Series:
        """Return strategy value history up to ``now``."""
        return self._history_to_now(self.data["value"])

    @property
    def capital(self) -> float:
        """Return currently unallocated capital."""
        return self._capital

    @property
    def cash(self) -> pd.Series:
        """Return cash history up to ``now``."""
        return self._history_to_now(self.data["cash"])

    @property
    def fees(self) -> pd.Series:
        """Return fee history up to ``now``."""
        return self._history_to_now(self.data["fees"])

    @property
    def flows(self) -> pd.Series:
        """Return flow history up to ``now``."""
        return self._history_to_now(self.data["flows"])

    @property
    def universe(self) -> pd.DataFrame:
        """Return the strategy's visible universe through ``now``."""
        return self._history_to_now(self._universe)

    @property
    def securities(self) -> list[Any]:
        """Return all descendant security nodes."""
        return [member for member in self.members if member._issec]

    @property
    def outlays(self) -> pd.DataFrame:
        """Return transaction outlays aggregated by security."""
        outlays = pd.DataFrame()
        for security in self.securities:
            if security.name in outlays.columns:
                outlays[security.name] += security.outlays
            else:
                outlays[security.name] = security.outlays
        return outlays

    @property
    def positions(self) -> pd.DataFrame:
        """Return security positions aggregated by security name."""
        vals = pd.DataFrame()
        for security in self.securities:
            if security.name in vals.columns:
                vals[security.name] += security.positions
            else:
                vals[security.name] = security.positions
        self._positions = vals
        return vals

    def setup(
        self,
        prices: pd.DataFrame,
        *,
        live_start_date: pd.Timestamp | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the strategy against a price universe."""
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame.")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices index must be a DatetimeIndex.")

        if live_start_date is None:
            live_start_ts = prices.index[0]
        else:
            live_start_ts = pd.Timestamp(live_start_date)
            if live_start_ts not in prices.index:
                raise ValueError("live_start_date must be present in prices index.")

        self._original_data = prices
        self._setup_kwargs = kwargs.copy()
        self._setup_kwargs.setdefault(
            "last_valid_date",
            prices.apply(lambda series: series.last_valid_index()),
        )
        self._live_start_date = live_start_ts

        universe = prices.copy()
        for child_name in self._strat_children:
            if child_name not in universe.columns:
                universe[child_name] = np.nan
        self._universe = pd.DataFrame(universe)

        self.bankrupt = False
        self.data = pd.DataFrame(
            index=self._universe.loc[live_start_ts:].index,
            columns=[
                "price",
                "value",
                "cash",
                "fees",
                "flows",
            ],
            data=0.0,
        )

        for child in self._childrenv:
            child.setup(prices, live_start_date=live_start_ts, **kwargs)

    def get_data(self, key: str) -> pd.DataFrame:
        """Return a setup-time auxiliary dataset by key."""
        return self._setup_kwargs[key]

    def _ensure_child(self, child_name: str) -> Node:
        """Create a direct security child when one does not yet exist."""
        if child_name in self.children:
            return self.children[child_name]

        child = Security(child_name)
        self._add_children([(child_name, child)])
        child = self.children[child_name]

        if hasattr(self, "_original_data"):
            child.setup(
                self._original_data,
                live_start_date=self._live_start_date,
                **self._setup_kwargs,
            )
            if self.now != 0:
                child.pre_market_update(self.now, self.inow)
                child._seed_pre_market_price()

        return child

    def _update_value(self) -> None:
        """Recompute strategy value from capital and child state."""
        self._value = self._capital

        for child in self._childrenv:
            self._value += child._value

    def _update_child_weights(self) -> None:
        """Recompute child weights from the current strategy value."""
        for child in self._childrenv:
            child._weight = (
                child._value / self._value if not is_zero(self._value) else 0.0
            )

    def _is_inactive_bankrupt_child(self) -> bool:
        """Return whether this strategy is a non-root strategy past its bankruptcy day."""
        return (
            self.parent is not None
            and self.bankrupt
            and self._bankrupt_at is not None
            and self.now != self._bankrupt_at
        )

    def pre_market_update(self, date: Any, inow: int) -> None:
        """Advance the strategy tree into the pre-market phase."""
        self._last_price = self._price
        self._last_value = self._value
        self._last_fee = 0.0
        self.last_day = self.now
        self.now = date
        self.inow = int(inow)

        if self._is_inactive_bankrupt_child():
            self._capital = 0.0
            self._value = 0.0
            self._weight = 0.0
            self._price = np.nan
            self._net_flows = 0.0
            return

        for child in self._childrenv:
            child.pre_market_update(date, self.inow)

    def post_market_update(self) -> None:
        """Finalize state after the market close using current close marks."""
        if self._is_inactive_bankrupt_child():
            self._capital = 0.0
            self._value = 0.0
            self._weight = 0.0
            self._price = np.nan
            self.data.loc[self.now, ["price", "value", "cash", "fees", "flows"]] = (
                np.nan
            )
            for child_name in self._strat_children:
                self._universe.loc[self.now, child_name] = np.nan
            self._net_flows = 0.0
            return

        for child in self._childrenv:
            child.post_market_update()
        self._update_value()
        self._update_child_weights()

        if self._value < -1e-12 and not self.bankrupt:
            self.bankrupt = True
            self._bankrupt_at = self.now
            self.perm["closed"].add(self.name)

        bottom = self._last_value + self._net_flows
        if not is_zero(bottom):
            ret = self._value / bottom - 1.0
        elif is_zero(self._value):
            ret = 0.0
        elif is_zero(self._value - self._net_flows, tol=1e-8):
            ret = 0.0
        else:
            raise ZeroDivisionError(
                "Could not update %s on %s. Last value was %s and net flows were %s. "
                "Current value is %s."
                % (
                    self.name,
                    self.now,
                    self._last_value,
                    self._net_flows,
                    self._value,
                )
            )

        self._price = self._last_price * (1.0 + ret)
        self.data.at[self.now, "price"] = self._price

        self.data.at[self.now, "value"] = self._value

        self.data.at[self.now, "cash"] = self._capital
        self.data.at[self.now, "fees"] = self._last_fee
        self.data.at[self.now, "flows"] = self._net_flows
        for child_name in self._strat_children:
            self._universe.loc[self.now, child_name] = self.children[child_name].price
        self._net_flows = 0.0

    def adjust(
        self,
        amount: float,
        flow: bool = True,
        fee: float = 0.0,
    ) -> None:
        """Adjust strategy capital by ``amount``."""
        self._capital += amount
        self._last_fee += fee

        if flow:
            self._net_flows += amount

    def allocate(
        self,
        amount: float,
        child: str | None = None,
    ) -> None:
        """Allocate capital to a child or down the strategy tree."""
        if self.bankrupt:
            return

        if is_zero(amount):
            return

        if child is not None:
            self._ensure_child(child).allocate(amount)
            return

        if self.parent is not None:
            self.parent.adjust(-amount, flow=False)
            self.adjust(amount, flow=True)
        self._update_value()

        for child_node in self._childrenv:
            child_node.allocate(amount * child_node._weight)

    def rebalance(
        self,
        weight: float,
        child: str,
        base: float,
    ) -> None:
        """Rebalance ``child`` to ``weight`` of ``base`` capital."""
        if is_zero(weight):
            if child in self.children:
                self.close(child)
            return

        current_child = self._ensure_child(child)
        delta = weight - current_child._weight
        current_child.allocate(delta * base)

    def close(self, child: str) -> None:
        """Close an existing child position."""
        if child not in self.children:
            return

        current_child = self.children[child]
        if current_child.children:
            current_child.flatten()

        if current_child._value != 0.0 and not np.isnan(current_child._value):
            current_child.allocate(-current_child._value)

    def flatten(self) -> None:
        """Close all child positions."""
        for child in self._childrenv:
            if child._value != 0.0 and not np.isnan(child._value):
                child.allocate(-child._value)

    def set_commissions(self, fn) -> None:
        """Set the commission function recursively on strategy children."""
        self.commission_fn = fn

        for child in self._childrenv:
            if getattr(child, "_is_strategy", False):
                child.set_commissions(fn)

    def run(self) -> None:
        """Run the algo stack, then run child nodes."""
        if self.bankrupt:
            self.temp = {}
            return

        self.temp = {}

        self.stack(self)

        for child in self._childrenv:
            child.run()
