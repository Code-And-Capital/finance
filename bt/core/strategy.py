import pandas as pd
import numpy as np
from copy import deepcopy
from bt.core.nodes import Node
from bt.core.algo_base import AlgoStack
from bt.core.security import SecurityBase, Security
from utils.math_utils import is_zero


class StrategyBase(Node):
    """
    Strategy Node. Defines capital allocation logic inside a strategy tree.

    A Strategy's primary job is to allocate capital to its children based on
    user-defined logic implemented in the `run()` method. Strategies may hold
    capital, delegate capital to children, track prices and values over time,
    and maintain references to a data universe provided during backtesting.

    Parameters
    ----------
    name : str
        Name of the strategy.
    children : dict | list | None
        Children of this strategy. If dict: {name: child_object}. If list:
        child objects. Children may be Nodes or strings. String values are
        lazily created when first referenced.
    parent : Node | None
        Parent node in the tree.

    Attributes
    ----------
    name : str
        Strategy name.
    parent : StrategyBase
        Parent strategy.
    root : StrategyBase
        Root (topmost) node in tree.
    children : dict
        Child nodes, keyed by name.
    now : datetime
        Current timestamp during backtesting.
    stale : bool
        Whether the strategy requires updating before reading values.
    prices, values, notional_values : pd.Series
        Historical strategy metrics over time.
    capital : float
        Unallocated capital (cash).
    commission_fn : callable
        Function computing transaction commissions.
    universe : pd.DataFrame
        Filtered data universe for this strategy.
    """

    def __init__(self, name: str, children=None, parent=None):
        super().__init__(name, children=children, parent=parent)

        self._weight = 1.0
        self._value = 0.0
        self._notl_value = 0.0
        self._price = 100

        # helper variables used in update calculations
        self._net_flows = 0.0
        self._last_value = 0.0
        self._last_notl_value = 0.0
        self._last_price = 100
        self._last_fee = 0.0

        self._last_chk = 0

        # default commission function (can be overridden)
        self.commission_fn = self._dflt_comm_fn

        # internal trading state
        self._paper_trade = False
        self._positions = None
        self.bankrupt = False

    # ----------------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------------

    @property
    def price(self):
        """Return the current strategy price, updating the tree if stale."""
        if self.root.stale:
            self.root.update(self.now, None)
        return self._price

    @property
    def prices(self):
        """Return the historical TimeSeries of strategy prices."""
        if self.root.stale:
            self.root.update(self.now, None)
        return self._prices.loc[: self.now]

    @property
    def values(self):
        """Return the historical TimeSeries of strategy values."""
        if self.root.stale:
            self.root.update(self.now, None)
        return self._values.loc[: self.now]

    @property
    def notional_values(self):
        """Return the TimeSeries of notional values."""
        if self.root.stale:
            self.root.update(self.now, None)
        return self._notl_values.loc[: self.now]

    @property
    def capital(self):
        """Return unallocated capital remaining in the strategy."""
        return self._capital

    @property
    def cash(self):
        """Return the historical TimeSeries of unallocated capital."""
        return self._cash

    @property
    def fees(self):
        """Return the historical TimeSeries of fees."""
        if self.root.stale:
            self.root.update(self.now, None)
        return self._fees.loc[: self.now]

    @property
    def flows(self):
        """Return the TimeSeries of capital flows (allocations, deposits, etc.)."""
        if self.root.stale:
            self.root.update(self.now, None)
        return self._all_flows.loc[: self.now]

    @property
    def bidoffer_paid(self):
        """
        Return bid/offer spread paid in the current step.

        Raises
        ------
        Exception
            If bid/offer accounting is not enabled.
        """
        if not self._bidoffer_set:
            raise Exception(
                'Bid/offer accounting not enabled. Provide "bidoffer" argument in setup().'
            )

        if self.root.stale:
            self.root.update(self.now, None)

        return self._bidoffer_paid

    @property
    def bidoffers_paid(self):
        """
        Return historical bid/offer spreads paid.

        Raises
        ------
        Exception
            If bid/offer accounting is not enabled.
        """
        if not self._bidoffer_set:
            raise Exception(
                'Bid/offer accounting not enabled. Provide "bidoffer" argument in setup().'
            )

        if self.root.stale:
            self.root.update(self.now, None)

        return self._bidoffers_paid.loc[: self.now]

    @property
    def universe(self):
        """
        Return the filtered universe up to the current timestamp.

        Memoized for the current date to avoid repeated slicing.
        """
        if self.now == self._last_chk:
            return self._funiverse

        self._last_chk = self.now
        self._funiverse = self._universe.loc[: self.now]
        return self._funiverse

    @property
    def securities(self):
        """Return a list of child nodes that represent securities."""
        return [x for x in self.members if isinstance(x, SecurityBase)]

    @property
    def outlays(self):
        """Return a DataFrame containing outlays for each security child."""
        if self.root.stale:
            self.root.update(self.root.now, None)

        outlays = pd.DataFrame()
        for sec in self.securities:
            col = sec.name
            outlays[col] = outlays.get(col, 0) + sec.outlays

        return outlays

    @property
    def positions(self):
        """Return the current position TimeSeries for all security children."""
        if self.root.stale:
            self.root.update(self.root.now, None)

        df = pd.DataFrame()
        for sec in self.securities:
            df[sec.name] = sec.positions

        self._positions = df.fillna(0.0)
        return self._positions

    def setup(self, universe: pd.DataFrame, **kwargs) -> None:
        """
        Initialize the strategy using the provided universe.

        Prepares internal data structures, validates fixed-income constraints,
        sets up child nodes recursively, and constructs the filtered universe
        used for strategy logic.

        Parameters
        ----------
        universe : pd.DataFrame
            Raw universe data used during backtesting.
        **kwargs
            Additional parameters, including optional "bidoffer".
        """
        # Save original universe and kwargs for later use (child setup, lazy children)
        self._original_data = universe
        self._setup_kwargs = kwargs.copy()

        # fixed income guard: cannot nest fixed-income beneath non-fixed-income
        if getattr(self, "fixed_income", False) and not getattr(
            self.parent, "fixed_income", False
        ):
            raise ValueError(
                "Cannot have fixed income strategy child (%s) of non-fixed income strategy (%s)"
                % (self.name, self.parent.name)
            )

        # If this node is not the parent (i.e., dynamically created child),
        # create a paper-trade clone to simulate allocations without mutating real tree
        if self is not self.parent:
            self._paper_trade = True
            self._paper_amount = 1000000

            paper = deepcopy(self)
            paper.parent = paper
            paper.root = paper
            paper._paper_trade = False
            # Use the same original data and kwargs for setup of the paper copy
            paper.setup(self._original_data, **kwargs)
            paper.adjust(self._paper_amount)
            self._paper = paper

        # create a working copy of universe for this strategy
        funiverse = universe.copy()

        # If original children were provided, restrict universe to those tickers
        if getattr(self, "_original_children_are_present", False):
            valid_filter = list(
                set(universe.columns).intersection(
                    self._universe_tickers + list(self._lazy_children.keys())
                )
            )
            funiverse = universe[valid_filter].copy()

            # If strategy children exist, add their columns to funiverse
            if getattr(self, "_has_strat_children", False):
                for child_name in self._strat_children:
                    funiverse[child_name] = np.nan

            # ensure it's a DataFrame to avoid pandas warnings
            funiverse = pd.DataFrame(funiverse)

        # store universe-related structures
        self._universe = funiverse
        self._funiverse = funiverse
        self._last_chk = None

        # reset bankruptcy flag
        self.bankrupt = False

        # Initialize internal time-series storage with zeros
        self.data = pd.DataFrame(
            index=funiverse.index,
            columns=["price", "value", "notional_value", "cash", "fees", "flows"],
            data=0.0,
        )

        self._prices = self.data["price"]
        self._values = self.data["value"]
        self._notl_values = self.data["notional_value"]
        self._cash = self.data["cash"]
        self._fees = self.data["fees"]
        self._all_flows = self.data["flows"]

        # Bid/offer accounting if requested
        if "bidoffer" in kwargs:
            self._bidoffer_set = True
            self.data["bidoffer_paid"] = 0.0
            self._bidoffers_paid = self.data["bidoffer_paid"]
        else:
            self._bidoffer_set = False

        # Setup children recursively using the original (unfiltered) universe
        if self.children is not None:
            for c in self._childrenv:
                c.setup(universe, **kwargs)

    def setup_from_parent(self, **kwargs) -> None:
        """
        Setup a strategy using parent's saved setup context.

        Used when dynamically creating child strategies.

        Parameters
        ----------
        **kwargs
            Additional arguments that will be merged with parent's setup kwargs.
        """
        all_kwargs = self.parent._setup_kwargs.copy()
        all_kwargs.update(kwargs)
        self.setup(self.parent._original_data, **all_kwargs)
        if self.name not in self.parent._universe:
            self.parent._universe[self.name] = np.nan

    def get_data(self, key: str):
        """
        Retrieve additional data that was provided to setup via kwargs.

        Parameters
        ----------
        key : str
            Name of the data item to retrieve.

        Returns
        -------
        object
            The data object passed at backtest creation under `key`.

        Raises
        ------
        KeyError
            If the key is not present in setup kwargs.
        """
        return self._setup_kwargs[key]

    def update(self, date, data=None, inow: int = None) -> None:
        """
        Update strategy state for the provided date.

        This updates prices, values, notional values, children weights, and
        associated time-series entries. It also handles bankruptcy detection
        and paper-trade logic when enabled.

        Parameters
        ----------
        date : Timestamp
            Date to update to.
        data : optional
            Additional data payload for the update (unused by default).
        inow : int | None
            Index location within the time-series for `date`. If None, it will
            be derived from the index.
        """
        # mark tree as not stale (we are recomputing now)
        self.root.stale = False

        # Determine if this is a new point in time (newpt)
        newpt = False
        if self.now == 0:
            newpt = True
        elif date != self.now:
            self._net_flows = 0.0
            self._last_price = self._price
            self._last_value = self._value
            self._last_notl_value = self._notl_value
            self._last_fee = 0.0
            newpt = True

        # update now
        self.now = date
        if inow is None:
            if self.now == 0:
                inow = 0
            else:
                inow = self.data.index.get_loc(date)

        # Calculate aggregate values by updating children
        val = getattr(self, "_capital", 0.0)  # default if no children
        notl_val = 0.0
        bidoffer_paid = 0.0
        coupons = 0.0

        if self.children:
            for c in self._childrenv:
                # Collect coupons from security children at each new point in time
                if getattr(c, "_issec", False) and newpt:
                    coupons += c._capital
                    c._capital = 0.0

                # Skip update if security and not flagged for update
                if getattr(c, "_issec", False) and not getattr(c, "_needupdate", True):
                    continue

                c.update(date, data, inow)
                val += c.value
                notl_val += abs(c.notional_value)

                if self._bidoffer_set:
                    bidoffer_paid += c.bidoffer_paid

        # adding coupon cash back to capital and total value
        self._capital = getattr(self, "_capital", 0.0) + coupons
        val += coupons

        # Bankruptcy check (only apply to root)
        if self.root == self:
            if (
                (val < 0)
                and (not self.bankrupt)
                and (not getattr(self, "fixed_income", False))
                and (not is_zero(val))
            ):
                self.bankrupt = True
                self.flatten()

        # Only update series when something meaningful changed or a new time point occurred
        if (
            newpt
            or (not is_zero(self._value - val))
            or (not is_zero(self._notl_value - notl_val))
        ):
            self._value = val
            self._values.values[inow] = val

            self._notl_value = notl_val
            self._notl_values.values[inow] = notl_val

            if self._bidoffer_set:
                self._bidoffer_paid = bidoffer_paid
                self._bidoffers_paid.values[inow] = bidoffer_paid

            if getattr(self, "fixed_income", False):
                # For fixed-income strategies compute additive return on notional
                pnl = self._value - (self._last_value + self._net_flows)
                if not is_zero(self._last_notl_value):
                    ret = pnl / self._last_notl_value * 100
                elif not is_zero(self._notl_value):
                    # occurs when building an initial position and last notional was zero
                    ret = pnl / self._notl_value * 100
                else:
                    if is_zero(pnl):
                        ret = 0.0
                    else:
                        raise ZeroDivisionError(
                            "Could not update %s on %s. Last notional value was %s and pnl was %s."
                            % (self.name, self.now, self._last_notl_value, pnl)
                        )
                self._price = self._last_price + ret
                self._prices.values[inow] = self._price
            else:
                bottom = self._last_value + self._net_flows
                if not is_zero(bottom):
                    ret = self._value / bottom - 1.0
                else:
                    if is_zero(self._value):
                        ret = 0.0
                    else:
                        raise ZeroDivisionError(
                            "Could not update %s on %s. Last value was %s and net flows were %s. Current value is %s."
                            % (
                                self.name,
                                self.now,
                                self._last_value,
                                self._net_flows,
                                self._value,
                            )
                        )
                self._price = self._last_price * (1.0 + ret)
                self._prices.values[inow] = self._price

        # Update child weights based on new totals
        if self.children:
            for c in self._childrenv:
                if getattr(c, "_issec", False) and not getattr(c, "_needupdate", True):
                    continue

                if getattr(self, "fixed_income", False):
                    c._weight = (
                        (c.notional_value / notl_val)
                        if (not is_zero(notl_val))
                        else 0.0
                    )
                else:
                    c._weight = (c.value / val) if (not is_zero(val)) else 0.0

        # If there are strategy children, reflect their price into our universe
        if getattr(self, "_has_strat_children", False):
            for c in self._strat_children:
                self._universe.loc[date, c] = self.children[c].price

        # Always update cash, fees, and flows for the date
        self._cash.values[inow] = self._capital
        self._fees.values[inow] = self._last_fee
        self._all_flows.values[inow] = self._net_flows

        # Update paper trade copy if enabled
        if self._paper_trade:
            if newpt:
                self._paper.update(date)
                self._paper.run()
                self._paper.update(date)
            # sync price with paper simulation
            self._price = self._paper.price
            self._prices.values[inow] = self._price

    def adjust(
        self, amount: float, update: bool = True, flow: bool = True, fee: float = 0.0
    ) -> None:
        """
        Adjust capital for this strategy (e.g., deposit or withdraw cash).

        Parameters
        ----------
        amount : float
            Amount to add (positive) or remove (negative).
        update : bool, default True
            If True, mark the root as stale (so values will be recomputed).
        flow : bool, default True
            If True, treat the adjustment as a capital flow (does not affect returns).
        fee : float, default 0.0
            Fee to record alongside the adjustment (affects performance if non-flow).
        """
        # Adjust capital and last fee
        self._capital += amount
        self._last_fee += fee

        # Flows are capital injections/withdrawals that shouldn't affect returns
        if flow:
            self._net_flows += amount

        if update:
            self.root.stale = True

    def allocate(self, amount: float, child: str = None, update: bool = True) -> None:
        """
        Allocate capital from this strategy.

        If `child` is provided, allocate only to that child (creating it lazily
        if necessary). Otherwise, allocate to this strategy: deduct from parent
        and distribute to children according to their weights.

        Parameters
        ----------
        amount : float
            Amount of capital to allocate.
        child : str | None
            Name of child to allocate to; if None allocate to this node.
        update : bool, default True
            If True, mark the root stale after allocation.
        """
        if child is not None:
            # ensure child exists (lazy create)
            self._create_child_if_needed(child)
            self.children[child].allocate(amount)
            return

        # Adjust parent's capital to offset allocation
        if self.parent == self:
            # if parent is self (root), treat allocation as a flow for the parent
            self.parent.adjust(-amount, update=False, flow=True)
        else:
            # parent is a strategy, do not mark that as a flow (affects performance)
            self.parent.adjust(-amount, update=False, flow=False)

        # Add capital to this node
        self.adjust(amount, update=False, flow=True)

        # Distribute down to children proportionally by internal _weight
        if self.children is not None:
            for c in self._childrenv:
                c.allocate(amount * c._weight, update=False)

        if update:
            self.root.stale = True

    def transact(self, q: float, child: str = None, update: bool = True) -> None:
        """
        Transact a notional amount across the strategy (used for fixed-income).

        If `child` is provided, the notional `q` is directed to that child;
        otherwise it is spread to children proportionally to their weights.

        Parameters
        ----------
        q : float
            Notional quantity to transact.
        child : str | None
            If provided, transact into that named child.
        update : bool, default True
            If True, mark the root stale after transaction.
        """
        if child is not None:
            self._create_child_if_needed(child)
            self.children[child].transact(q)
            return

        if self.children is not None:
            for c in self._childrenv:
                c.transact(q * c._weight, update=False)

        if update:
            self.root.stale = True

    def rebalance(
        self, weight: float, child: str, base: float = np.nan, update: bool = True
    ) -> None:
        """
        Rebalance a specific child's target weight.

        For fixed-income strategies, this will use notional transact calls;
        for normal strategies it will allocate delta capital.

        Parameters
        ----------
        weight : float
            Target weight for the child (typically between -1.0 and 1.0).
        child : str
            Name of the child to rebalance.
        base : float, optional
            Base value to compute weight changes from; if NaN uses the node's current
            value or notional_value depending on fixed_income.
        update : bool, default True
            If True, update the root's stale flag after operation.
        """
        # Closing the child when weight is effectively zero
        if is_zero(weight):
            if child in self.children:
                self.close(child, update=update)
            return

        # Determine base if not provided
        if np.isnan(base):
            base = self.notional_value if self.fixed_income else self.value

        # Ensure child exists
        self._create_child_if_needed(child)
        c = self.children[child]

        if self.fixed_income:
            # For fixed income strategies, weights refer to notional allocation
            delta = weight * base - c.weight * self.notional_value
            if c.fixed_income:
                c.transact(delta, update=update)
            else:
                c.allocate(delta, update=update)
        else:
            # Non fixed-income: weight is fraction of total value
            delta = weight - c.weight
            c.allocate(delta * base, update=update)

    def close(self, child: str, update: bool = True) -> None:
        """
        Close (flatten) a child's position.

        For fixed-income, uses transact to clear position; otherwise allocates
        a negative amount equal to the child's value to zero it.

        Parameters
        ----------
        child : str
            Child name to close.
        update : bool, default True
            Whether to mark the root stale after the operation.
        """
        c = self.children[child]
        # Flatten child's children first
        if c.children is not None and len(c.children) != 0:
            c.flatten()

        if self.fixed_income:
            if getattr(c, "position", 0.0) != 0.0:
                c.transact(-c.position, update=update)
        else:
            if (getattr(c, "value", 0.0) != 0.0) and (not np.isnan(c.value)):
                c.allocate(-c.value, update=update)

    def flatten(self) -> None:
        """
        Close out all child positions for this strategy.

        Equivalent to closing every child. Marks the root stale.
        """
        if getattr(self, "fixed_income", False):
            for c in self._childrenv:
                if getattr(c, "position", 0) != 0:
                    c.transact(-c.position, update=False)
        else:
            for c in self._childrenv:
                if getattr(c, "value", 0) != 0:
                    c.allocate(-c.value, update=False)

        self.root.stale = True

    def run(self) -> None:
        """
        Main algorithm entry point. Override this method with strategy logic.

        This method is called by the backtester at each date change and should
        contain the decision-making logic for allocations, rebalances, trades, etc.
        """
        pass  # user-defined logic

    def set_commissions(self, fn) -> None:
        """
        Set commission (transaction fee) function for the strategy tree.

        The provided function should accept (quantity, price) and return a fee
        amount. The function will be applied recursively to strategy children.

        Parameters
        ----------
        fn : callable
            Function of signature fn(quantity, price) -> fee_amount
        """
        self.commission_fn = fn
        for c in self._childrenv:
            if isinstance(c, StrategyBase):
                c.set_commissions(fn)

    def get_transactions(self) -> pd.DataFrame:
        """
        Build a MultiIndex DataFrame of transactions.

        Returned DataFrame format:
            index: (Date, Security)
            columns: ['price', 'quantity']

        The method determines trades by differencing security positions and
        aligns them with historical prices. If bid/offer adjustments are
        enabled they are included in the price.

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame of transactions (date, security).
        """
        # Prices for each security (unstacked)
        prc = pd.DataFrame({x.name: x.prices for x in self.securities}).unstack()

        # Aggregate positions per security
        positions = pd.DataFrame()
        for x in self.securities:
            if x.name in positions.columns:
                positions[x.name] += x.positions
            else:
                positions[x.name] = x.positions

        # Trades are the diff of positions; first row is the first position
        trades = positions.diff()
        if len(positions) > 0:
            trades.iloc[0] = positions.iloc[0]

        # Convert to long form and drop zero trades
        trades = trades[trades != 0].unstack().dropna()

        # Adjust prices for bid/offer paid if set
        if getattr(self, "_bidoffer_set", False):
            bidoffer = pd.DataFrame(
                {x.name: x.bidoffers_paid for x in self.securities}
            ).unstack()
            # Avoid division by zero by aligning indices; trades is non-zero here
            prc = prc + (bidoffer / trades)

        res = pd.DataFrame({"price": prc, "quantity": trades}).dropna(
            subset=["quantity"]
        )
        res.index.names = ["Security", "Date"]
        res = res.swaplevel().sort_index()
        return res

    def _dflt_comm_fn(self, q: float, p: float) -> float:
        """Default commission function (no fees)."""
        return 0.0

    def _create_child_if_needed(self, child: str) -> None:
        """
        Ensure the named child exists; if not, create lazily from _lazy_children
        or create a default Security node.

        After creating the child the method sets it up and updates it to the
        current date.
        """
        if child not in self.children:
            # pop a lazily defined child or create a default security
            c = self._lazy_children.pop(child, Security(child))
            c.lazy_add = False

            # add child to tree without calling data-cleanup (dc=False)
            self._add_children([c], dc=False)
            c.setup(self._universe, **self._setup_kwargs)

            # bring the new child up-to-date
            c.update(self.now)


class Strategy(StrategyBase):
    """
    A Strategy represents a composite algorithmic node that manages capital
    allocation logic by executing a stack of Algos and optionally delegating
    execution to child Strategy nodes.

    This class extends :class:`StrategyBase` by incorporating an :class:`AlgoStack`.
    A Strategy is constructed by passing a list of Algos, which are wrapped into
    an AlgoStack and executed in sequence whenever ``run()`` is called.

    Two shared data containers are exposed to allow information exchange between
    Algos during execution:

    - ``temp``: Temporary, per-run data. Cleared at the start of each ``run()``
      call. Use this for intermediate or transient values needed only within
      a single pass.
    - ``perm``: Persistent, cross-run state that is preserved across multiple
      executions. Use this for values that must flow between Algos over time.

    After running its Algo stack, the Strategy will recursively run its child
    Strategy nodes (if any), allowing for hierarchical strategy design.

    Parameters
    ----------
    name : str
        The name of the strategy.
    algos : list of Algo, optional
        A list of Algo instances that define the strategy's logic. If omitted, an
        empty stack is used.
    children : dict or list, optional
        Child nodes of this Strategy. Accepts either:
        - A dict mapping names to Node instances.
        - A list of child nodes or names (string names result in lazy creation
          of Node instances when accessed).
    parent : Node, optional
        The parent node in a larger strategy tree.

    Attributes
    ----------
    stack : AlgoStack
        The ordered stack of Algos executed when ``run()`` is called.
    temp : dict
        Temporary data cleared at the start of each run.
    perm : dict
        Persistent data preserved across runs of the Strategy.
    """

    def __init__(self, name, algos=None, children=None, parent=None):
        super().__init__(name, children=children, parent=parent)
        if algos is None:
            algos = []
        self.stack = AlgoStack(*algos)
        self.temp = {}
        self.perm = {}

    def run(self):
        """
        Execute the strategy by running all Algos in the AlgoStack, then
        recursively executing all child nodes.

        The process is:
        1. Clear temporary data.
        2. Execute the AlgoStack, allowing Algos to read/write ``temp`` and ``perm``.
        3. Run each child Strategy or Node in sequence.

        Returns
        -------
        None
        """
        # clear out temp data
        self.temp = {}

        # run algo stack
        self.stack(self)

        # run children
        for c in self._childrenv:
            c.run()
