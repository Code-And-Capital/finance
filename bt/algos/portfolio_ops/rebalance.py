from bt.algos.core import Algo
import numpy as np
import pandas as pd
from utils.math_utils import validate_integer


class Rebalance(Algo):
    """Rebalance strategy positions to target weights.

    This algo applies ``temp["weights"]`` to the strategy, closes open
    positions not present in the target mapping, and then calls
    ``target.rebalance(...)`` for each target weight.

    Notes
    -----
    - This implementation is equity-only and does not include fixed-income
      notional branches.
    - Optional ``temp["cash"]`` reserves a fraction of portfolio value from
      allocation (for example, ``cash=0.2`` keeps 20% unallocated).
    """

    def __init__(self) -> None:
        """
        Initialize Rebalance algorithm.
        """
        super().__init__()

    def __call__(self, target) -> bool:
        """Execute one full rebalance step.

        Returns
        -------
        bool
            ``True`` on successful execution. Returns ``False`` only when
            required input shapes are malformed (for example non-dict weights
            or invalid cash fraction).
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return False

        if "weights" not in temp:
            return True

        raw_targets = temp["weights"]
        if isinstance(raw_targets, pd.Series):
            targets = raw_targets.dropna().to_dict()
        elif isinstance(raw_targets, dict):
            targets = raw_targets
        else:
            return False

        base = target.value

        cash_fraction_raw = temp.get("cash", 0.0)
        try:
            cash_fraction = float(cash_fraction_raw)
        except (TypeError, ValueError):
            return False
        if cash_fraction < 0.0 or cash_fraction > 1.0:
            return False
        base *= 1.0 - cash_fraction

        # Close children not present in target mapping.
        for cname, child in target.children.items():
            if cname in targets:
                continue

            v = child.value
            if v != 0.0 and not np.isnan(v):
                target.close(cname, update=False)

        # Apply target allocations.
        for child_name, weight in targets.items():
            target.rebalance(weight, child=child_name, base=base, update=False)

        # Propagate tree state once after all child operations.
        target.root.update(target.now)

        return True


class RebalanceOverTime(Algo):
    """Apply target weights gradually over ``n`` calls.

    When ``temp["weights"]`` is present, the algo snapshots it as a destination
    and applies linear partial moves from current weights to destination over
    ``n`` steps. Each step delegates execution to :class:`Rebalance`.
    """

    def __init__(self, n: int = 10):
        """Initialize the multi-step rebalance scheduler."""
        super().__init__()
        self.run_always = True
        n_steps = int(validate_integer(n, "RebalanceOverTime `n`"))
        if n_steps <= 0:
            raise ValueError("RebalanceOverTime `n` must be > 0.")

        self.n = n_steps
        self._rb = Rebalance()
        self._weights: dict | None = None
        self._days_left: int | None = None

    def _start_schedule(self, raw_weights) -> bool:
        """Start a new schedule from provided destination weights."""
        if isinstance(raw_weights, pd.Series):
            self._weights = raw_weights.dropna().to_dict()
        elif isinstance(raw_weights, dict):
            self._weights = raw_weights
        else:
            return False
        self._days_left = self.n
        return True

    def _clear_schedule(self) -> None:
        """Clear any active schedule state."""
        self._weights = None
        self._days_left = None

    def __call__(self, target):
        """Run one schedule step (or initialize schedule from new weights)."""
        temp = self._resolve_temp(target)
        if temp is None:
            return False

        if "weights" in temp:
            if not self._start_schedule(temp["weights"]):
                return False

        if self._weights is not None:
            if self._days_left is None or self._days_left <= 0:
                self._clear_schedule()
                return True

            tgt: dict[str, float] = {}
            for cname, target_weight in self._weights.items():
                current_weight = (
                    target.children[cname].weight if cname in target.children else 0.0
                )
                delta = (target_weight - current_weight) / self._days_left
                tgt[cname] = current_weight + delta

            temp["weights"] = tgt
            if not self._rb(target):
                return False

            self._days_left -= 1
            if self._days_left <= 0:
                self._clear_schedule()

        return True
