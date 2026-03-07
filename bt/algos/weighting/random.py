from __future__ import annotations

from typing import Any

import numpy as np
import drs

from bt.algos.weighting.core import WeightAlgo
from bt.algos.weighting.optimizers.base_optimizer import BaseOptimizer
from bt.algos.weighting.optimizers.validators import validate_bounds


class RandomWeightOptimizer(BaseOptimizer):
    """Optimizer that samples feasible random weights under box bounds."""

    def __init__(
        self,
        bounds: tuple[float, float] = (0.0, 1.0),
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.bounds = validate_bounds(bounds, "RandomWeightOptimizer")
        self.random_seed = random_seed

    def set_problem(
        self,
        selected: list[str],
    ) -> None:
        """Store selected assets for random-weight solve."""
        self.reset()
        assets = list(set(selected))
        n_assets = len(assets)
        lower, upper = self.bounds
        min_weights = np.full(n_assets, lower, dtype=float)
        max_weights = np.full(n_assets, upper, dtype=float)
        self.set_problem_data(
            assets=assets,
            asset_count=n_assets,
            min_weights=min_weights,
            max_weights=max_weights,
        )

    def solve_problem(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Solve random-weight allocation and return standard result payload."""
        assets = self.problem_data.get("assets", [])
        asset_count = self.problem_data.get("asset_count", 0)
        min_weights = self.problem_data.get("min_weights")
        max_weights = self.problem_data.get("max_weights")

        if asset_count == 0:
            allocations: dict[str, float] = {}
        elif asset_count == 1:
            allocations = {assets[0]: 1.0}
        else:
            target = 1.0
            min_total = asset_count * self.bounds[0]
            max_total = asset_count * self.bounds[1]
            if target < min_total or target > max_total:
                raise ValueError(
                    "RandomWeightOptimizer infeasible problem: "
                    "n_assets*bounds[0] <= 1 <= n_assets*bounds[1] must hold."
                )
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            # TODO: Replace DRS with convolutionalfixedsum once installation/build is stable.
            sampled = np.asarray(
                drs.drs(asset_count, 1, max_weights, min_weights),
                dtype=float,
            ).reshape(-1)
            allocations = {
                asset: float(weight) for asset, weight in zip(assets, sampled)
            }

        self.weights_ = allocations
        self.success = True
        self.status = "optimal"
        self.message = "Solved analytically."
        return self.get_result()


class WeightRandomly(WeightAlgo):
    """Assign bounded random weights for names in ``temp['selected']``."""

    def __init__(
        self,
        bounds: tuple[float, float] = (0.0, 1.0),
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.optimizer = RandomWeightOptimizer(
            bounds=bounds,
            random_seed=random_seed,
        )

    def __call__(self, target: Any) -> bool:
        """Compute and store random weights in ``temp['weights']``."""
        temp = self._resolve_temp(target)
        if temp is None:
            return False
        now = self._resolve_now(target)

        selected_raw = temp.get("selected", [])
        if not isinstance(selected_raw, list):
            return False
        if not selected_raw:
            self._write_weights(temp, {})
            self._record_allocation_history(now, {})
            return True
        if len(selected_raw) == 1:
            weights = {selected_raw[0]: 1.0}
            self._write_weights(temp, weights)
            self._record_allocation_history(now, weights)
            return True

        self.optimizer.set_problem(selected_raw)
        result = self.optimizer.solve_problem()
        weights = result["weights"]
        self._write_weights(temp, weights)
        self._record_allocation_history(now, weights)
        return True
