from __future__ import annotations

from typing import Any

import cvxpy as cvx
import numpy as np
import pandas as pd

from bt.algos.weighting.core import WeightAlgo
from bt.algos.weighting.optimizers.constraints import (
    bound_constraints,
    sum_to_one_constraint,
)
from bt.algos.weighting.optimizers.convex_optimizer import ConvexOptimizer
from bt.algos.weighting.optimizers.objectives import min_variance_objective
from bt.algos.weighting.optimizers.validators import (
    resolve_selected_covariance,
    validate_bounds,
    validate_square_covariance_matrix,
)
from bt.algos.weighting.optimizers.variables import (
    build_covariance_matrix,
    build_weight_variable,
)


class MinVarianceOptimizer(ConvexOptimizer):
    """Convex minimum-variance optimizer with simple box bounds."""

    def __init__(self, bounds: tuple[float, float] = (0.0, 1.0)) -> None:
        super().__init__()
        self.bounds = validate_bounds(bounds, "MinVarianceOptimizer")

    def set_problem(
        self,
        cov: pd.DataFrame,
        selected: list[str],
        **kwargs: Any,
    ) -> None:
        """Set covariance inputs and assemble objective/constraints."""
        self.reset()
        validate_square_covariance_matrix(cov, "MinVarianceOptimizer")
        cov = resolve_selected_covariance(cov, selected)

        asset_count = len(selected)
        if asset_count == 0:
            self.set_problem_data(asset_count=0, universe=[], weights_var=None)
            self._constraints = []
            self._objective = lambda: cvx.Maximize(0)
            return

        cov_matrix = build_covariance_matrix(cov)
        weights = build_weight_variable(asset_count, self.variable)
        min_weights, max_weights = self.compute_weight_bounds(selected, self.bounds)

        self.add_objective(
            lambda: min_variance_objective(
                weights,
                cov_matrix,
                self.minimize,
            )
        )
        self.bulk_add_constraints(bound_constraints(weights, min_weights, max_weights))
        self.add_constraint(sum_to_one_constraint(weights))
        self.set_problem_data(
            universe=selected,
            asset_count=asset_count,
            weights_var=weights,
        )

    def solve_problem(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Solve minimum-variance problem and return weights payload."""
        asset_count = int(self.problem_data.get("asset_count", 0))
        universe = self.problem_data.get("universe", [])
        weights_var = self.problem_data.get("weights_var")
        if asset_count == 0:
            self.weights_ = {}
            self.success = True
            self.status = "optimal"
            self.message = "No assets to optimize."
            return self.get_result()
        if asset_count == 1:
            self.weights_ = {universe[0]: 1.0}
            self.success = True
            self.status = "optimal"
            self.message = "Single asset."
            return self.get_result()

        super().solve_problem(*args, **kwargs)
        if weights_var is None or weights_var.value is None:
            raise RuntimeError("MinVarianceOptimizer solved without weights.")
        solved = np.asarray(weights_var.value, dtype=float).reshape(-1)
        self.weights_ = {
            asset: float(weight) for asset, weight in zip(universe, solved)
        }
        return self.get_result()


class WeightMinVar(WeightAlgo):
    """Assign portfolio weights using convex minimum-variance optimization."""

    def __init__(
        self,
        bounds: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.bounds = bounds
        self.optimizer = MinVarianceOptimizer(bounds=self.bounds)

    def __call__(self, target: Any) -> bool:
        """Compute and store minimum-variance weights in ``temp['weights']``."""
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

        cov = temp.get("covariance")
        self.optimizer.set_problem(cov, selected_raw)
        result = self.optimizer.solve_problem()
        weights = result["weights"]
        self._write_weights(temp, weights)
        self._record_allocation_history(now, weights)
        return True
