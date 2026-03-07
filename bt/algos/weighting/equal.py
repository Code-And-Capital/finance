from __future__ import annotations

from typing import Any

from bt.algos.weighting.core import WeightAlgo
from bt.algos.weighting.optimizers.base_optimizer import BaseOptimizer


class EqualWeightOptimizer(BaseOptimizer):
    """Analytical optimizer for equal-weight portfolios.

    Given a selected universe, each asset receives ``1 / n_assets``.
    """

    def __init__(self) -> None:
        super().__init__()

    def set_problem(self, universe: list[Any]) -> None:
        """Store the selected universe for solving.

        Parameters
        ----------
        universe
            Candidate asset names.
        """
        self.reset()
        if not isinstance(universe, list):
            raise TypeError("EqualWeightOptimizer `universe` must be a list.")
        deduped_universe = list(set(universe))
        self.set_problem_data(universe=deduped_universe, n_assets=len(deduped_universe))

    def solve_problem(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Compute equal weights and return a standard optimizer payload."""
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
    """Assign equal weights across ``temp['selected']``.

    Behavior
    --------
    - Reads ``target.temp['selected']``.
    - Writes computed weights to ``target.temp['weights']``.
    - Records allocation history for each run.
    """

    def __init__(self) -> None:
        """Initialize assigner and backing analytical optimizer."""
        super().__init__()
        self.optimizer = EqualWeightOptimizer()

    def __call__(self, target) -> bool:
        """Run equal-weight assignment for the current strategy state.

        Returns
        -------
        bool
            ``True`` when assignment was processed, ``False`` for invalid
            context (for example missing/invalid ``temp`` or ``selected``).
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return False

        selected_raw = temp.get("selected", [])
        if not isinstance(selected_raw, list):
            return False

        self.optimizer.set_problem(selected_raw)
        result = self.optimizer.solve_problem()
        allocations = result["weights"]
        now = self._resolve_now(target)
        self._write_weights(temp, allocations, now=now, record_history=True)
        return True
