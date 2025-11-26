from typing import Any
from bt.core.algo_base import Algo


class RunIfOutOfBounds(Algo):
    """
    Algo that triggers when any security weight deviates beyond a given tolerance.

    This is useful for strategies where rebalancing occurs either periodically
    or when the portfolio weights drift too far from targets.

    Examples
    --------
    Rebalance quarterly or when any security deviates by more than 20%:
        Or([RunQuarterly(), RunIfOutOfBounds(0.2)])

    Args:
        tolerance : float
            Maximum allowed deviation from the target weight before triggering.
            Expressed as a fraction (e.g., 0.2 = 20%).

    Requires:
        - `weights` stored in `target.temp`
    """

    def __init__(self, tolerance: float):
        """
        Initialize RunIfOutOfBounds.

        Parameters
        ----------
        tolerance : float
            Maximum deviation allowed for any security before triggering.
        """
        super().__init__()
        self.tolerance: float = float(tolerance)

    def __call__(self, target: Any) -> bool:
        """
        Determine if any security or cash weight is outside the tolerance.

        Parameters
        ----------
        target : bt.backtest.Target
            The backtest target object, which must include `temp['weights']`.

        Returns
        -------
        bool
            True if any weight or cash allocation exceeds the tolerance, False otherwise.
        """
        # If no weights are defined yet, allow algo to run
        if "weights" not in target.temp:
            return True

        targets = target.temp["weights"]

        # Check each child security
        for cname, target_weight in targets.items():
            if cname in target.children:
                current_weight = target.children[cname].weight
                deviation = abs((current_weight - target_weight) / target_weight)
                if deviation > self.tolerance:
                    return True

        # Optional: check cash deviation
        if "cash" in target.temp:
            cash_target = targets.value  # assumes targets.value includes total capital
            cash_deviation = abs(
                (target.capital - cash_target) / cash_target - target.temp["cash"]
            )
            if cash_deviation > self.tolerance:
                return True

        return False
