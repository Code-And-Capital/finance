import math
from typing import Any

import numpy as np
import pandas as pd

from bt.core.nodes import Node
from utils.math_utils import is_zero


class Security(Node):
    """
    Base class for tradable security nodes.

    A security tracks position, marked value, and transaction outlays against
    its parent strategy.
    """

    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__(name=name, children=None)

        self._price = 0.0
        self._value = 0.0
        self._weight = 0.0
        self._position = 0.0
        self._last_pos = 0.0
        self._issec = True
        self._outlay = 0.0

        self._prices = pd.Series(dtype=float)

    def _history_to_now(self, series: pd.Series) -> pd.Series:
        """Return a history slice through the current timestamp."""
        if self.now == 0:
            return series.iloc[0:0]
        return series.loc[: self.now]

    @staticmethod
    def _round_integer_quantity(q: float) -> int:
        """Round an integer trade quantity toward zero."""
        return math.floor(q) if q >= 0.0 else math.ceil(q)

    @staticmethod
    def _is_feasible_outlay(full_outlay: float, amount: float) -> bool:
        """Return whether a candidate trade stays within the requested budget."""
        if amount >= 0.0:
            return full_outlay <= amount
        return full_outlay >= amount

    def _seed_pre_market_price(self) -> None:
        """Seed a new security's reference price from the prior close."""
        seed_inow = max(int(self.inow) - 1, 0)
        self._price = float(self._prices.iat[seed_inow])

    @property
    def price(self) -> float:
        """Return the latest marked price."""
        return self._price

    @property
    def prices(self) -> pd.Series:
        """Return price history up to ``now``."""
        return self._history_to_now(self._prices)

    @property
    def values(self) -> pd.Series:
        """Return value history up to ``now``."""
        return self._history_to_now(self.data["value"])

    @property
    def position(self) -> float:
        """Return current position size."""
        return self._position

    @property
    def positions(self) -> pd.Series:
        """Return position history up to ``now``."""
        return self._history_to_now(self.data["position"])

    @property
    def outlays(self) -> pd.Series:
        """Return transaction outlay history up to ``now``."""
        return self._history_to_now(self.data["outlay"])

    def setup(
        self,
        prices: pd.DataFrame,
        *,
        live_start_date: pd.Timestamp | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the security against a price universe."""
        del kwargs
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame.")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices index must be a DatetimeIndex.")
        if self.name not in prices.columns:
            raise ValueError(f"Missing prices for security '{self.name}'.")

        if live_start_date is None:
            live_start_ts = prices.index[0]
        else:
            live_start_ts = pd.Timestamp(live_start_date)
            if live_start_ts not in prices.index:
                raise ValueError("live_start_date must be present in prices index.")

        self._prices = prices[self.name]
        self._price = float(self._prices.iat[0])
        self.data = pd.DataFrame(
            index=prices.loc[live_start_ts:].index,
            columns=["value", "position"],
            data=0.0,
        )
        self.data["outlay"] = 0.0

    def pre_market_update(self, date: Any, inow: int) -> None:
        """Advance the security into the pre-market phase without remarking prices."""
        self.last_day = self.now
        self.now = date
        self.inow = int(inow)

    def post_market_update(self) -> None:
        """Mark the security to the current close and persist the day."""
        self._price = float(self._prices.iat[self.inow])
        if np.isnan(self._price):
            if is_zero(self._position):
                self._value = 0.0
            else:
                raise RuntimeError(
                    f"Open position ({self._position}) but price is NaN for "
                    f"{self.name} on {self.now}."
                )
        else:
            self._value = self._position * self._price

        self.data.at[self.now, "position"] = self._position
        self.data.at[self.now, "value"] = self._value

        if self._outlay != 0.0:
            self.data.at[self.now, "outlay"] += self._outlay
            self._outlay = 0.0

        self._last_pos = self._position

    def allocate(self, amount: float) -> None:
        """Allocate capital to the security."""
        if is_zero(amount):
            return
        if is_zero(self._price) or np.isnan(self._price):
            raise RuntimeError(
                f"Cannot allocate capital to {self.name} because price is "
                f"{self._price} as of {self.now}."
            )

        if is_zero(amount + self._value):
            q = -self._position
        else:
            q = amount / self._price
            if self.integer_positions:
                q = self._round_integer_quantity(q)

        if is_zero(q) or np.isnan(q):
            return

        if q != -self._position:
            full_outlay, _, _ = self.outlay(q)
            last_q = q
            last_amount_short = full_outlay - amount

            for _ in range(10_000):
                if np.isclose(full_outlay, amount, rtol=1e-16) or q == 0:
                    break

                dq = (full_outlay - amount) / self._price
                q = q - dq

                if self.integer_positions:
                    q = self._round_integer_quantity(q)

                full_outlay, _, _ = self.outlay(q)

                if self.integer_positions:
                    neighbor_q = q + 1 if q >= 0 else q - 1
                    neighbor_outlay, _, _ = self.outlay(neighbor_q)
                    if (
                        min(full_outlay, neighbor_outlay)
                        <= amount
                        <= max(full_outlay, neighbor_outlay)
                    ):
                        if self._is_feasible_outlay(neighbor_outlay, amount) and (
                            not self._is_feasible_outlay(full_outlay, amount)
                            or abs(neighbor_outlay - amount) < abs(full_outlay - amount)
                        ):
                            q = neighbor_q
                            full_outlay = neighbor_outlay
                        break

                if self.integer_positions and last_q == q:
                    raise RuntimeError(
                        "Integer position adjustment stuck because quantity did not change."
                    )
                if np.abs(full_outlay - amount) > np.abs(last_amount_short):
                    raise RuntimeError(
                        "Quantity search moved further away from the requested amount."
                    )

                last_q = q
                last_amount_short = full_outlay - amount
            else:
                raise RuntimeError(
                    "Infinite loop detected while adjusting allocation quantity."
                )

        self.transact(q)

    def transact(
        self,
        q: float,
    ) -> None:
        """Buy or sell ``q`` units of the security."""
        if is_zero(q) or np.isnan(q):
            return

        self._position += q
        if not np.isnan(self._price):
            self._value = self._position * self._price

        full_outlay, outlay, fee = self.outlay(q)
        self._outlay += outlay

        self.parent.adjust(-full_outlay, flow=False, fee=fee)

    def commission(self, q: float, p: float) -> float:
        """Return the commission implied by the parent strategy."""
        if self.parent is None:
            raise RuntimeError("Cannot calculate commission for a parentless security.")
        return self.parent.commission_fn(q, p)

    def outlay(self, q: float) -> tuple[float, float, float]:
        """Return full outlay, base outlay, and fee."""
        execution_price = self._price
        outlay = q * execution_price
        fee = self.commission(q, execution_price)
        return outlay + fee, outlay, fee

    def run(self) -> None:
        """Securities do not execute strategy logic."""
        return None
