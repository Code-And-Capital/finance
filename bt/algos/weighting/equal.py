from __future__ import annotations

from typing import Any

from bt.algos.weighting.core import WeightAlgo
from bt.algos.weighting.optimizers.base_optimizer import BaseOptimizer


class EqualWeightOptimizer(BaseOptimizer):
    """Optimizer that solves equal-weight allocations."""

    def __init__(self) -> None:
        super().__init__()

    def set_problem(self, universe: list[Any]) -> None:
        """Set selected universe for equal-weight optimization."""
        self.reset()
        if not isinstance(universe, list):
            raise TypeError("EqualWeightOptimizer `universe` must be a list.")
        deduped_universe = list(set(universe))
        self.set_problem_data(universe=deduped_universe, n_assets=len(deduped_universe))

    def solve_problem(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Solve equal-weight allocation and return standard result payload."""
        universe = self.problem_data.get("universe", [])
        n_assets = self.problem_data.get("n_assets", 0)
        if not isinstance(universe, list) or not isinstance(n_assets, int):
            raise TypeError("EqualWeightOptimizer problem data is invalid.")

        if n_assets == 0:
            allocations: dict[Any, float] = {}
        else:
            weight = 1.0 / n_assets
            allocations = {asset: weight for asset in universe}

        self.weights_ = allocations
        self.success = True
        self.status = "optimal"
        self.message = "Solved analytically."
        return self.get_result()


class WeightEqually(WeightAlgo):
    """Assign equal weights across selected assets.

    This assigner delegates optimization to ``EqualWeightOptimizer`` and writes
    the resulting weights into ``target.temp['weights']``.
    """

    def __init__(self) -> None:
        """Initialize assigner and underlying optimizer."""
        super().__init__()
        self.optimizer = EqualWeightOptimizer()

    def __call__(self, target) -> bool:
        """Compute and store equal weights into ``target.temp['weights']``."""
        temp = self._resolve_temp(target)
        if temp is None:
            return False

        selected_raw = temp.get("selected", [])
        if not isinstance(selected_raw, list):
            return False

        self.optimizer.set_problem(selected_raw)
        result = self.optimizer.solve_problem()
        allocations = result["weights"]
        self._write_weights(temp, allocations)

        now = self._resolve_now(target)
        self._record_allocation_history(now, allocations)
        return True
