from bt.core import Algo
import numpy as np


class Rebalance(Algo):
    """
    Rebalances capital according to temp['weights'].

    This algorithm allocates capital based on temp['weights'] and closes
    positions that are open but not in the target weights. It is typically
    the last algorithm called after all selection and weighting algos
    have been applied.

    Requires:
        temp['weights']: dict of target weights per security
        temp['cash'] (optional): float between 0-1 specifying fraction
            of capital to hold as cash. Default is 0 (fully invested).
        temp['notional_value'] (optional): for fixed-income portfolios,
            base notional value for weighting calculations.
    """

    def __init__(self) -> None:
        """
        Initialize Rebalance algorithm.
        """
        super().__init__()

    def __call__(self, target) -> bool:
        """
        Rebalance capital and close positions not in target weights.

        Parameters:
            target: Strategy/backtest object providing:
                - children: dict of child nodes (securities)
                - temp: dict of temporary variables (weights, cash, etc.)
                - fixed_income: bool, indicates fixed-income portfolio
                - value: total equity
                - notional_value: total notional value for fixed-income
                - close(child_name, update=False): method to close positions
                - rebalance(weight, child=child_name, base=base, update=False): method to allocate
                - root.update(now): method to propagate updates
                - now: current date

        Returns:
            bool: Always True
        """
        if "weights" not in target.temp:
            return True

        targets = target.temp["weights"]

        # Determine base for allocations
        if target.fixed_income:
            base = target.temp.get("notional_value", target.notional_value)
        else:
            base = target.value

        # Close children that are not in target weights and have non-zero value
        for cname, child in target.children.items():
            if cname in targets:
                continue

            v = child.notional_value if target.fixed_income else child.value
            if v != 0.0 and not np.isnan(v):
                target.close(cname, update=False)

        # Adjust base for cash allocation if applicable
        if "cash" in target.temp and not target.fixed_income:
            base *= 1 - target.temp["cash"]

        # Allocate according to target weights
        for child_name, weight in targets.items():
            target.rebalance(weight, child=child_name, base=base, update=False)

        # Update after rebalancing
        target.root.update(target.now)

        return True


class RebalanceOverTime(Algo):
    """
    Gradually rebalance the portfolio toward a new target weight vector over
    multiple periods.

    This Algo smooths rebalancing by dividing the weight adjustment into `n`
    equal steps applied over consecutive periods. This is useful when the
    strategy's raw weight changes are large, and a full instantaneous rebalance
    would be unrealistic. For example, in monthly strategies the rebalance may
    be spread across several trading days.

    When a new `temp['weights']` dictionary is provided, the Algo initializes a
    multi-period schedule and applies one portion of the weight change at each
    call. Once all steps are completed, the schedule resets.

    In typical monthly strategies, this usually requires the `run_always`
    decorator, because many RunPeriod algos return `False` after the first
    day of the period.

    Requires:
        * weights: A dictionary of {security: target_weight} that will trigger
          the start of a new multi-period rebalance sequence.

    Parameters:
        * n (int): Number of periods over which to apply the rebalance.

    Returns:
        * bool: Always returns True.
    """

    def __init__(self, n: int = 10):
        """
        Initialize a RebalanceOverTime instance.

        Parameters:
            * n (int): Number of periods over which the rebalance is applied.
              Larger values smooth the adjustment over more periods.
        """
        super().__init__()
        self.n = float(n)
        self._rb = Rebalance()
        self._weights = None
        self._days_left = None

    def __call__(self, target):
        """
        Execute one step of the multi-period rebalance.

        Behavior:
            * If new weights are present in temp['weights'], initialize a new
              rebalance schedule over `n` future periods.
            * On each call, compute the partial progress toward the target
              weights and apply it via the standard Rebalance Algo.
            * After `n` steps, reset all internal state.

        Parameters:
            * target: The AlgoTarget on which the rebalance is executed.

        Returns:
            * bool: Always returns True.
        """
        # New weights trigger a new rebalance schedule.
        if "weights" in target.temp:
            self._weights = target.temp["weights"]
            self._days_left = self.n

        # Continue scheduled rebalancing if active.
        if self._weights is not None:
            tgt = {}

            for cname in self._weights.keys():
                curr = (
                    target.children[cname].weight if cname in target.children else 0.0
                )
                dlt = (self._weights[cname] - curr) / self._days_left
                tgt[cname] = curr + dlt

            # Inject temporary weights and rebalance one step.
            target.temp["weights"] = tgt
            self._rb(target)

            # Update schedule state.
            self._days_left -= 1

            if self._days_left == 0:
                self._days_left = None
                self._weights = None

        return True
