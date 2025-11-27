from __future__ import annotations
from typing import Optional, Any, Dict, Union, Tuple
import numpy as np
import pandas as pd
from bt.core.nodes import Node
from utils.math_utils import is_zero
import math


class SecurityBase(Node):
    """
    Security Node. Represents a tradable asset in the tree.

    Attributes
    ----------
    name : str
        Security name.
    multiplier : float
        Multiplier for position size (e.g., derivatives, lots).
    lazy_add : bool
        Whether the security should be added lazily to parent strategy.
    _position : float
        Current position (quantity held).
    _price : float
        Current price.
    _value : float
        Current value of the security.
    _weight : float
        Weight in parent.
    _bidoffer : float
        Current bid/offer spread.
    _needupdate : bool
        Flag indicating whether security needs update.
    """

    def __init__(
        self, name: str, multiplier: float = 1, lazy_add: bool = False
    ) -> None:
        super().__init__(name, parent=None, children=None)
        self._value: float = 0
        self._price: float = 0
        self._weight: float = 0
        self._position: float = 0
        self.multiplier: float = multiplier
        self.lazy_add: bool = lazy_add

        self._last_pos: float = 0
        self._issec: bool = True
        self._needupdate: bool = True
        self._outlay: float = 0
        self._bidoffer: float = 0

    @property
    def price(self) -> float:
        """Current security price. Updates if stale."""
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        return self._price

    @property
    def prices(self) -> pd.Series:
        """TimeSeries of prices up to current time."""
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        return self._prices.loc[: self.now]

    @property
    def values(self) -> pd.Series:
        """TimeSeries of values up to current time."""
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._values.loc[: self.now]

    @property
    def notional_values(self) -> pd.Series:
        """TimeSeries of notional values."""
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._notl_values.loc[: self.now]

    @property
    def position(self) -> float:
        """Current position (quantity held)."""
        return self._position

    @property
    def positions(self) -> pd.Series:
        """TimeSeries of positions."""
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._positions.loc[: self.now]

    @property
    def outlays(self) -> pd.Series:
        """TimeSeries of capital outlays for transactions."""
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._outlays.loc[: self.now]

    @property
    def bidoffer(self) -> float:
        """Current bid/offer spread."""
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        return self._bidoffer

    @property
    def bidoffers(self) -> pd.Series:
        """TimeSeries of bid/offer spreads."""
        if self._bidoffer_set:
            if self._needupdate or self.now != self.parent.now:
                self.update(self.root.now)
            return self._bidoffers.loc[: self.now]
        raise RuntimeError(
            'Bid/offer accounting not enabled. Provide "bidoffer" during setup.'
        )

    @property
    def bidoffer_paid(self) -> float:
        """Bid/offer paid in current step."""
        if self._needupdate or self.now != self.parent.now:
            self.update(self.root.now)
        return self._bidoffer_paid

    @property
    def bidoffers_paid(self) -> pd.Series:
        """TimeSeries of bid/offer paid on transactions."""
        if self._bidoffer_set:
            if self._needupdate or self.now != self.parent.now:
                self.update(self.root.now)
            if self.root.stale:
                self.root.update(self.root.now, None)
            return self._bidoffers_paid.loc[: self.now]
        raise RuntimeError(
            'Bid/offer accounting not enabled. Provide "bidoffer" during setup.'
        )

    def setup(self, universe: pd.DataFrame, **kwargs: Any) -> None:
        """
        Initialize Security with price data and optional bid/offer spreads.

        Parameters
        ----------
        universe : pd.DataFrame
            Price data for this security.
        kwargs : dict
            Optional additional data, e.g., 'bidoffer' DataFrame.
        """
        try:
            prices = universe[self.name]
        except KeyError:
            prices = None

        if prices is not None:
            self._prices = prices
            self.data = pd.DataFrame(
                index=universe.index,
                columns=["value", "position", "notional_value"],
                data=0.0,
            )
            self._prices_set = True
        else:
            self.data = pd.DataFrame(
                index=universe.index,
                columns=["price", "value", "position", "notional_value"],
            )
            self._prices = self.data["price"]
            self._prices_set = False

        self._values = self.data["value"]
        self._notl_values = self.data["notional_value"]
        self._positions = self.data["position"]

        self.data["outlay"] = 0.0
        self._outlays = self.data["outlay"]

        if "bidoffer" in kwargs:
            self._bidoffer_set = True
            self._bidoffers = kwargs["bidoffer"]
            bidoffers = self._bidoffers.get(self.name, None)
            if bidoffers is not None:
                if not bidoffers.index.equals(universe.index):
                    raise ValueError("Bid/offer index must match universe index.")
                self._bidoffers = bidoffers
            else:
                self.data["bidoffer"] = 0.0
                self._bidoffers = self.data["bidoffer"]
            self.data["bidoffer_paid"] = 0.0
            self._bidoffers_paid = self.data["bidoffer_paid"]

    def update(
        self, date: Any, data: Optional[pd.DataFrame] = None, inow: Optional[int] = None
    ) -> None:
        """Update security values, positions, and bid/offer state for a given date."""
        if date == self.now and self._last_pos == self._position:
            return

        if inow is None:
            inow = 0 if date == 0 else self.data.index.get_loc(date)

        if date != self.now:
            self.now = date
            if self._prices_set:
                self._price = self._prices.values[inow]
            elif data is not None:
                self._price = data[self.name]
                self._prices.values[inow] = self._price
            if self._bidoffer_set:
                self._bidoffer = self._bidoffers.values[inow]
                self._bidoffer_paid = 0.0

        self._positions.values[inow] = self._position
        self._last_pos = self._position

        if np.isnan(self._price):
            if is_zero(self._position):
                self._value = 0
            else:
                raise RuntimeError(
                    f"Open position ({self._position}) but price is NaN for {self.name} on {date}."
                )
        else:
            self._value = self._position * self._price * self.multiplier

        self._notl_value = self._value
        self._values.values[inow] = self._value
        self._notl_values.values[inow] = self._notl_value

        if is_zero(self._weight) and is_zero(self._position):
            self._needupdate = False

        if self._outlay != 0:
            self._outlays.values[inow] += self._outlay
            self._outlay = 0

        if self._bidoffer_set:
            self._bidoffers_paid.values[inow] = self._bidoffer_paid

    def allocate(self, amount: float, update: bool = True) -> None:
        """
        Allocate capital to this security (buy/sell).

        Determines the quantity to transact given the current price,
        taking into account integer positions and commission/bid-offer costs.

        Parameters
        ----------
        amount : float
            Capital amount to allocate (positive to buy, negative to sell).
        update : bool
            Whether to update parent after allocation.
        """
        if self._needupdate or self.now != self.parent.now:
            self.update(self.parent.now)

        if is_zero(amount):
            return

        if self.parent is self or self.parent is None:
            raise RuntimeError("Cannot allocate capital to a parentless security")

        if is_zero(self._price) or np.isnan(self._price):
            raise RuntimeError(
                f"Cannot allocate capital to {self.name} because price is {self._price} as of {self.parent.now}"
            )

        # Compute desired quantity
        if is_zero(amount + self._value):
            q = -self._position  # closing position
        else:
            q = amount / (self._price * self.multiplier)
            if self.integer_positions:
                q = (
                    math.floor(q)
                    if (self._position > 0 or amount > 0)
                    else math.ceil(q)
                )

        if is_zero(q) or np.isnan(q):
            return

        # Newton-like adjustment for integer positions and commission
        if not q == -self._position:
            full_outlay, _, _, _ = self.outlay(q)

            i = 0
            last_q = q
            last_amount_short = full_outlay - amount
            while not np.isclose(full_outlay, amount, rtol=1e-16) and q != 0:
                dq_wout_considering_tx_costs = (full_outlay - amount) / (
                    self._price * self.multiplier
                )
                q = q - dq_wout_considering_tx_costs

                if self.integer_positions:
                    q = math.floor(q)

                full_outlay, _, _, _ = self.outlay(q)

                if self.integer_positions:
                    full_outlay_of_1_more, _, _, _ = self.outlay(q + 1)

                    if full_outlay < amount and full_outlay_of_1_more > amount:
                        break

                i = i + 1
                if i > 1e4:
                    raise RuntimeError(
                        "Infinite loop detected while adjusting allocation quantity"
                    )

                if self.integer_positions and last_q == q:
                    raise RuntimeError(
                        "Integer position adjustment stuck, q did not change."
                    )

                last_q = q

                if np.abs(full_outlay - amount) > np.abs(last_amount_short):
                    raise RuntimeError(
                        "The difference between what we have raised with q and"
                        " the amount we are trying to raise has gotten bigger since"
                        " last iteration! full_outlay should always be approaching"
                        " amount! There may be a case where the commission fn is"
                        " not smooth"
                    )
                last_amount_short = full_outlay - amount

        self.transact(q, update, update_self=False)

    def transact(
        self,
        q: float,
        update: bool = True,
        update_self: bool = True,
        price: Optional[float] = None,
    ) -> None:
        """
        Transact the security (buy/sell a given quantity).

        Parameters
        ----------
        q : float
            Quantity to transact.
        update : bool
            Whether to update parent with the resulting cash adjustment.
        update_self : bool
            Whether to update this security before transacting.
        price : float, optional
            Custom price for the transaction (e.g., bid/ask).
        """
        if update_self and (self._needupdate or self.now != self.parent.now):
            self.update(self.parent.now)

        if is_zero(q) or np.isnan(q):
            return

        if price is not None and not self._bidoffer_set:
            raise ValueError(
                "Cannot transact at custom prices without bid/offer tracking enabled."
            )

        self._needupdate = True
        self._position += q

        full_outlay, outlay, fee, bidoffer = self.outlay(q, p=price)
        self._outlay += outlay
        self._bidoffer_paid += bidoffer

        self.parent.adjust(-full_outlay, update=update, flow=False, fee=fee)

    def commission(self, q: float, p: float) -> float:
        """
        Compute transaction commission based on parent's commission function.

        Parameters
        ----------
        q : float
            Quantity to transact.
        p : float
            Price per unit.

        Returns
        -------
        float
            Total commission fee.
        """
        return self.parent.commission_fn(q, p)

    def outlay(
        self, q: float, p: Optional[float] = None
    ) -> Tuple[float, float, float, float]:
        """
        Calculate full cash outlay including bid/offer and commission.

        Parameters
        ----------
        q : float
            Quantity to transact.
        p : float, optional
            Custom transaction price.

        Returns
        -------
        Tuple[float, float, float, float]
            Full outlay, base outlay, commission fee, bid/offer adjustment.
        """
        if p is None:
            fee = self.commission(q, self._price * self.multiplier)
            bidoffer = abs(q) * 0.5 * self._bidoffer * self.multiplier
        else:
            fee = self.commission(q, p * self.multiplier)
            bidoffer = q * (p - self._price) * self.multiplier

        outlay = q * self._price * self.multiplier + bidoffer
        return outlay + fee, outlay, fee, bidoffer

    def run(self) -> None:
        """Securities do nothing on run."""
        pass


class Security(SecurityBase):
    """
    A standard security node with no special features.

    Notional value is measured based on market value (quantity * price * multiplier).

    This class exists primarily to distinguish standard securities from
    nonstandard ones. For example:

        isinstance(sec, Security)  # True only for vanilla securities
        isinstance(sec, SecurityBase)  # True for all securities

    Attributes
    ----------
    Inherits all attributes and methods from SecurityBase.
    """

    pass


class FixedIncomeSecurity(SecurityBase):
    """
    Fixed Income Security Node.

    Notional value is based solely on the position size (par value) of the
    security, rather than market value.

    Typically used within a FixedIncomeStrategy.

    Attributes
    ----------
    Inherits all attributes and methods from SecurityBase.
    Notable difference: `_notl_value` is equal to `_position` instead of
    market value.
    """

    def update(self, date, data=None, inow=None) -> None:
        """
        Update the security for a given date and optionally some data.

        This method updates price, value, weight, and also adjusts
        the notional value to be equal to the position size.

        Parameters
        ----------
        date : any
            The current date or time step for the update.
        data : optional
            Optional external data used to update prices.
        inow : int, optional
            Index in the internal DataFrame corresponding to `date`.
        """
        if inow is None:
            if date == 0:
                inow = 0
            else:
                inow = self.data.index.get_loc(date)

        # Call base class update
        super().update(date, data, inow)

        # Fixed income: notional value = position size
        self._notl_value = self._position
        self._notl_values.values[inow] = self._notl_value


class HedgeSecurity(SecurityBase):
    """
    Hedge Security Node.

    A security where the notional value is always set to zero, meaning it does not
    contribute to the notional value of the strategy.

    Typically used in fixed income strategies for hedging instruments
    (e.g., treasury bonds or interest rate swaps) that should not count
    toward the portfolio's notional exposure.

    Attributes
    ----------
    Inherits all attributes and methods from SecurityBase.
    Notable difference: `_notl_value` is always 0.
    """

    def update(self, date, data=None, inow=None) -> None:
        """
        Update the security for a given date and optionally some data.

        This updates price, value, weight, etc., but always sets the notional
        value to zero.

        Parameters
        ----------
        date : any
            The current date or time step for the update.
        data : optional
            Optional external data used to update prices.
        inow : int, optional
            Index in the internal DataFrame corresponding to `date`.
        """
        # Call base update
        super().update(date, data, inow)

        # Hedge: notional value is always zero
        self._notl_value = 0.0
        self._notl_values.values.fill(0.0)


class CouponPayingSecurity(FixedIncomeSecurity):
    """
    Coupon-paying security.

    Extends FixedIncomeSecurity to handle securities that pay
    (possibly irregular) coupons or other forms of cash disbursement.
    Holding costs can also be accounted for. Coupons and costs
    are passed during setup.

    Attributes
    ----------
    _coupon : float
        Current coupon payment (position scaled).
    _holding_cost : float
        Current holding cost (position scaled).
    _coupons : pd.Series
        Time series of coupon payments.
    _cost_long : pd.Series
        Time series of long position holding costs.
    _cost_short : pd.Series
        Time series of short position holding costs.
    _coupon_income : pd.Series
        Stores coupon payments in internal DataFrame.
    _holding_costs : pd.Series
        Stores holding costs in internal DataFrame.
    """

    def __init__(
        self,
        name: str,
        multiplier: float = 1,
        fixed_income: bool = True,
        lazy_add: bool = False,
    ) -> None:
        super().__init__(name, multiplier)
        self._coupon: float = 0.0
        self._holding_cost: float = 0.0
        self._fixed_income: bool = fixed_income
        self.lazy_add: bool = lazy_add

    def setup(self, universe: pd.DataFrame, **kwargs) -> None:
        """
        Setup security with universe and coupon/cost data.

        Parameters
        ----------
        universe : pd.DataFrame
            DataFrame of prices with the security name as one column.
        coupons : pd.DataFrame
            Mandatory coupon/carry data with same index as universe.
        cost_long : pd.DataFrame, optional
            Long position holding costs.
        cost_short : pd.DataFrame, optional
            Short position holding costs.
        kwargs : dict
            Additional security-level information.
        """
        super().setup(universe, **kwargs)

        # Coupon handling
        if "coupons" not in kwargs:
            raise ValueError('"coupons" must be provided in setup.')

        self._coupons = kwargs["coupons"].get(self.name)
        if self._coupons is None or not self._coupons.index.equals(universe.index):
            raise ValueError("Coupons index must match universe data.")

        # Holding costs
        self._cost_long = kwargs.get("cost_long", {}).get(self.name)
        self._cost_short = kwargs.get("cost_short", {}).get(self.name)

        # Internal tracking
        self.data["coupon"] = 0.0
        self.data["holding_cost"] = 0.0
        self._coupon_income = self.data["coupon"]
        self._holding_costs = self.data["holding_cost"]

    def update(self, date, data=None, inow=None) -> None:
        """
        Update security for a given date, including coupon and holding costs.

        Parameters
        ----------
        date : any
            Current date or time step.
        data : optional
            External data for update.
        inow : int, optional
            Index corresponding to date in internal DataFrame.
        """
        if inow is None:
            inow = 0 if date == 0 else self.data.index.get_loc(date)

        if self._coupons is None:
            raise ValueError(f"Coupons not set for security {self.name}.")

        # Standard price/value update
        super().update(date, data, inow)

        coupon_value = self._coupons.values[inow]
        if np.isnan(coupon_value):
            self._coupon = 0.0 if is_zero(self._position) else np.nan
        else:
            self._coupon = self._position * coupon_value

        if self._position > 0 and self._cost_long is not None:
            self._holding_cost = self._position * self._cost_long.values[inow]
        elif self._position < 0 and self._cost_short is not None:
            self._holding_cost = -self._position * self._cost_short.values[inow]
        else:
            self._holding_cost = 0.0

        # Adjust capital
        self._capital = self._coupon - self._holding_cost
        self._coupon_income.values[inow] = self._coupon
        self._holding_costs.values[inow] = self._holding_cost

    @property
    def coupon(self) -> float:
        """Current coupon payment (position scaled)."""
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._coupon

    @property
    def coupons(self) -> pd.Series:
        """TimeSeries of coupons paid (position scaled)."""
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._coupon_income.loc[: self.now]

    @property
    def holding_cost(self) -> float:
        """Current holding cost (position scaled)."""
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._holding_cost

    @property
    def holding_costs(self) -> pd.Series:
        """TimeSeries of holding costs (position scaled)."""
        if self.root.stale:
            self.root.update(self.root.now, None)
        return self._holding_costs.loc[: self.now]


class CouponPayingHedgeSecurity(CouponPayingSecurity):
    """
    Coupon-paying hedge security.

    Extends CouponPayingSecurity where the notional value is always zero.
    Intended for fixed income strategies, e.g., hedges such as interest
    rate swaps or treasury bonds that should not count toward notional value.
    """

    def update(self, date, data=None, inow: int | None = None) -> None:
        """
        Update security for a given date, including coupon and holding costs.

        Parameters
        ----------
        date : any
            Current date or time step.
        data : optional
            External data for update.
        inow : int, optional
            Index corresponding to date in internal DataFrame.
        """
        super().update(date, data, inow)

        # Hedge securities have zero notional value
        self._notl_value = 0.0
        self._notl_values.values.fill(0.0)
