import numpy as np
import pandas as pd
import sklearn.covariance

import bt
from bt.core import Algo, AlgoStack, SecurityBase, is_zero


def run_always(f):
    """
    Run always decorator to be used with Algo
    to ensure stack runs the decorated Algo
    on each pass, regardless of failures in the stack.
    """
    f.run_always = True
    return f


class PTE_Rebalance(Algo):
    """
    Triggers a rebalance when PTE from static weights is past a level.

    Args:
        * PTE_volatility_cap: annualized volatility to target
        * target_weights: dataframe of weights that needs to have the same index as the price dataframe
        * lookback (DateOffset): lookback period for estimating volatility
        * lag (DateOffset): amount of time to wait to calculate the covariance
        * covar_method: method of calculating volatility
        * annualization_factor: number of periods to annualize by.
          It is assumed that target volatility is already annualized by this factor.

    """

    def __init__(
        self,
        PTE_volatility_cap,
        target_weights,
        lookback=pd.DateOffset(months=3),
        lag=pd.DateOffset(days=0),
        covar_method="standard",
        annualization_factor=252,
    ):
        super(PTE_Rebalance, self).__init__()
        self.PTE_volatility_cap = PTE_volatility_cap
        self.target_weights = target_weights
        self.lookback = lookback
        self.lag = lag
        self.covar_method = covar_method
        self.annualization_factor = annualization_factor

    def __call__(self, target):
        if target.now is None:
            return False

        if target.positions.shape == (0, 0):
            return True

        positions = target.positions.loc[target.now]
        if positions is None:
            return True
        prices = target.universe.loc[target.now, positions.index]
        if prices is None:
            return True

        current_weights = positions * prices / target.value

        target_weights = self.target_weights.loc[target.now, :]

        cols = list(current_weights.index.copy())
        for c in target_weights.keys():
            if c not in cols:
                cols.append(c)

        weights = pd.Series(np.zeros(len(cols)), index=cols)
        for c in cols:
            if c in current_weights:
                weights[c] = current_weights[c]
            if c in target_weights:
                weights[c] -= target_weights[c]

        t0 = target.now - self.lag
        prc = target.universe.loc[t0 - self.lookback : t0, cols]
        returns = bt.ffn.to_returns(prc)

        # calc covariance matrix
        if self.covar_method == "ledoit-wolf":
            covar = sklearn.covariance.ledoit_wolf(returns)
        elif self.covar_method == "standard":
            covar = returns.cov()
        else:
            raise NotImplementedError("covar_method not implemented")

        PTE_vol = np.sqrt(
            np.matmul(weights.values.T, np.matmul(covar.values, weights.values))
            * self.annualization_factor
        )

        if pd.isnull(PTE_vol):
            return False
        # vol is too high
        if PTE_vol > self.PTE_volatility_cap:
            return True
        else:
            return False

        return True


class Rebalance(Algo):
    """
    Rebalances capital based on temp['weights']

    Rebalances capital based on temp['weights']. Also closes
    positions if open but not in target_weights. This is typically
    the last Algo called once the target weights have been set.

    Requires:
        * weights
        * cash (optional): You can set a 'cash' value on temp. This should be a
          number between 0-1 and determines the amount of cash to set aside.
          For example, if cash=0.3, the strategy will allocate 70% of its
          value to the provided weights, and the remaining 30% will be kept
          in cash. If this value is not provided (default), the full value
          of the strategy is allocated to securities.
        * notional_value (optional): Required only for fixed_income targets. This is the base
          balue of total notional that will apply to the weights.
    """

    def __init__(self):
        super(Rebalance, self).__init__()

    def __call__(self, target):
        if "weights" not in target.temp:
            return True

        targets = target.temp["weights"]

        # save value because it will change after each call to allocate
        # use it as base in rebalance calls
        # call it before de-allocation so that notional_value is correct
        if target.fixed_income:
            if "notional_value" in target.temp:
                base = target.temp["notional_value"]
            else:
                base = target.notional_value
        else:
            base = target.value

        # de-allocate children that are not in targets and have non-zero value
        # (open positions)
        for cname in target.children:
            # if this child is in our targets, we don't want to close it out
            if cname in targets:
                continue

            # get child and value
            c = target.children[cname]
            if target.fixed_income:
                v = c.notional_value
            else:
                v = c.value

            # if non-zero and non-null, we need to close it out
            if v != 0.0 and not np.isnan(v):
                target.close(cname, update=False)

        # If cash is set (it should be a value between 0-1 representing the
        # proportion of cash to keep), calculate the new 'base'
        if "cash" in target.temp and not target.fixed_income:
            base = base * (1 - target.temp["cash"])

        # Turn off updating while we rebalance each child
        for item in targets.items():
            target.rebalance(item[1], child=item[0], base=base, update=False)

        # Now update
        target.root.update(target.now)

        return True


class RebalanceOverTime(Algo):
    """
    Similar to Rebalance but rebalances to target
    weight over n periods.

    Rebalances towards a target weight over a n periods. Splits up the weight
    delta over n periods.

    This can be useful if we want to make more conservative rebalacing
    assumptions. Some strategies can produce large swings in allocations. It
    might not be reasonable to assume that this rebalancing can occur at the
    end of one specific period. Therefore, this algo can be used to simulate
    rebalancing over n periods.

    This has typically been used in monthly strategies where we want to spread
    out the rebalancing over 5 or 10 days.

    Note:
        This Algo will require the run_always wrapper in the above case. For
        example, the RunMonthly will return True on the first day, and
        RebalanceOverTime will be 'armed'. However, RunMonthly will return
        False the rest days of the month. Therefore, we must specify that we
        want to always run this algo.

    Args:
        * n (int): number of periods over which rebalancing takes place.

    Requires:
        * weights

    """

    def __init__(self, n=10):
        super(RebalanceOverTime, self).__init__()
        self.n = float(n)
        self._rb = Rebalance()
        self._weights = None
        self._days_left = None

    def __call__(self, target):
        # new weights specified - update rebalance data
        if "weights" in target.temp:
            self._weights = target.temp["weights"]
            self._days_left = self.n

        # if _weights are not None, we have some work to do
        if self._weights is not None:
            tgt = {}
            # scale delta relative to # of periods left and set that as the new
            # target
            for cname in self._weights.keys():
                curr = (
                    target.children[cname].weight if cname in target.children else 0.0
                )
                dlt = (self._weights[cname] - curr) / self._days_left
                tgt[cname] = curr + dlt

            # mock weights and call real Rebalance
            target.temp["weights"] = tgt
            self._rb(target)

            # dec _days_left. If 0, set to None & set _weights to None
            self._days_left -= 1

            if self._days_left == 0:
                self._days_left = None
                self._weights = None

        return True


def _get_unit_risk(security, data, index=None):
    try:
        unit_risks = data[security]
        unit_risk = unit_risks.values[index]
    except Exception:
        # No risk data, assume zero
        unit_risk = 0.0
    return unit_risk


class UpdateRisk(Algo):
    """
    Tracks a risk measure on all nodes of the strategy. To use this node, the
    ``additional_data`` argument on :class:`Backtest <bt.backtest.Backtest>` must
    have a "unit_risk" key. The value should be a dictionary, keyed
    by risk measure, of DataFrames with a column per security that is sensitive to that measure.

    Args:
        * name (str): the name of the risk measure (IR01, PVBP, IsIndustials, etc).
          The name must coincide with the keys of the dictionary passed to additional_data as the
          "unit_risk" argument.
        * history (int): The level of depth in the tree at which to track the time series of risk numbers.
          i.e. 0=no tracking, 1=first level only, etc. More levels is more expensive.

    Modifies:
        * The "risk" attribute on the target and all its children
        * If history==True, the "risks" attribute on the target and all its children

    """

    def __init__(self, measure, history=0):
        super(UpdateRisk, self).__init__(name="UpdateRisk>%s" % measure)
        self.measure = measure
        self.history = history

    def _setup_risk(self, target, set_history):
        """Setup risk attributes on the node in question"""
        target.risk = {}
        if set_history:
            target.risks = pd.DataFrame(index=target.data.index)

    def _setup_measure(self, target, set_history):
        """Setup a risk measure within the risk attributes on the node in question"""
        target.risk[self.measure] = np.nan
        if set_history:
            target.risks[self.measure] = np.nan

    def _set_risk_recursive(self, target, depth, unit_risk_frame):
        set_history = depth < self.history
        # General setup of risk on nodes
        if not hasattr(target, "risk"):
            self._setup_risk(target, set_history)
        if self.measure not in target.risk:
            self._setup_measure(target, set_history)

        if isinstance(target, bt.core.SecurityBase):
            # Use target.root.now as non-traded securities may not have been updated yet
            # and there is no need to update them here as we only use position
            index = unit_risk_frame.index.get_loc(target.root.now)
            unit_risk = _get_unit_risk(target.name, unit_risk_frame, index)
            if is_zero(target.position):
                risk = 0.0
            else:
                risk = unit_risk * target.position * target.multiplier
        else:
            risk = 0.0
            for child in target.children.values():
                self._set_risk_recursive(child, depth + 1, unit_risk_frame)
                risk += child.risk[self.measure]

        target.risk[self.measure] = risk
        if depth < self.history:
            target.risks.loc[target.now, self.measure] = risk

    def __call__(self, target):
        unit_risk_frame = target.get_data("unit_risk")[self.measure]
        self._set_risk_recursive(target, 0, unit_risk_frame)
        return True


class HedgeRisks(Algo):
    """
    Hedges risk measures with selected instruments.

    Make sure that the UpdateRisk algo has been called beforehand.

    Args:
        * measures (list): the names of the risk measures to hedge
        * pseudo (bool): whether to use the pseudo-inverse to compute
          the inverse Jacobian. If False, will fail if the number
          of selected instruments is not equal to the number of
          measures, or if the Jacobian is singular
        * strategy (StrategyBase): If provided, will hedge the risk
          from this strategy in addition to the risk from target.
          This is to allow separate tracking of hedged and unhedged
          performance. Note that risk_strategy must occur earlier than
          'target' in a depth-first traversal of the children of the root,
          otherwise hedging will occur before positions of risk_strategy are
          updated.
        * throw_nan (bool): Whether to throw on nan hedge notionals, rather
          than simply not hedging.

    Requires:
        * selected
    """

    def __init__(self, measures, pseudo=False, strategy=None, throw_nan=True):
        super(HedgeRisks, self).__init__()
        if len(measures) == 0:
            raise ValueError("Must pass in at least one measure to hedge")
        self.measures = measures
        self.pseudo = pseudo
        self.strategy = strategy
        self.throw_nan = throw_nan

    def _get_target_risk(self, target, measure):
        if not hasattr(target, "risk"):
            raise ValueError("risk not set up on target %s" % target.name)
        if measure not in target.risk:
            raise ValueError("measure %s not set on target %s" % (measure, target.name))
        return target.risk[measure]

    def __call__(self, target):
        securities = target.temp["selected"]

        # Get target risk
        target_risk = np.array(
            [self._get_target_risk(target, m) for m in self.measures]
        )
        if self.strategy is not None:
            # Add the target risk of the strategy to the risk of the target
            # (which contains existing hedges)
            target_risk += np.array(
                [self._get_target_risk(self.strategy, m) for m in self.measures]
            )
        # Turn target_risk into a column array
        target_risk = target_risk.reshape(len(self.measures), 1)

        # Get hedge risk as a Jacobian matrix
        data = []
        for m in self.measures:
            d = target.get_data("unit_risk").get(m)
            if d is None:
                raise ValueError(
                    "unit_risk for %s not present in temp on %s"
                    % (self.measure, target.name)
                )
            i = d.index.get_loc(target.now)
            data.append((i, d))

        hedge_risk = np.array(
            [[_get_unit_risk(s, d, i) for (i, d) in data] for s in securities]
        )

        # Get hedge ratios
        if self.pseudo:
            inv = np.linalg.pinv(hedge_risk).T
        else:
            inv = np.linalg.inv(hedge_risk).T
        notionals = np.matmul(inv, -target_risk).flatten()

        # Hedge
        for notional, security in zip(notionals, securities):
            if np.isnan(notional) and self.throw_nan:
                raise ValueError("%s has nan hedge notional" % security)
            target.transact(notional, security)
        return True
