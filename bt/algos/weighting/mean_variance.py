from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import cvxpy as cvx

from bt.algos.weighting.core import WeightAlgo
from bt.algos.weighting.optimizers.constraints import (
    bound_constraints,
    sum_to_one_constraint,
)
from bt.algos.weighting.optimizers.convex_optimizer import ConvexOptimizer
from bt.algos.weighting.optimizers.objectives import mean_variance_utility_objective
from bt.algos.weighting.optimizers.validators import (
    resolve_selected_covariance,
    validate_series,
    validate_square_covariance_matrix,
)
from bt.algos.weighting.optimizers.variables import (
    build_covariance_matrix,
    build_expected_returns_parameter,
    build_weight_variable,
)


class MeanVarianceOptimizer(ConvexOptimizer):
    """Convex mean-variance optimizer with optional turnover and group constraints."""

    def __init__(self) -> None:
        super().__init__()

    def set_problem(
        self,
        rets: pd.Series,
        cov: pd.DataFrame,
        selected: list[str],
        risk_averse_lambda: float = 1.0,
        bounds: tuple[float, float] = (0.0, 1.0),
        **kwargs: Any,
    ) -> None:
        """Set expected returns/covariance and assemble CVX objective+constraints."""
        self.reset()
        validate_series(rets, "MeanVarianceOptimizer", "rets")
        validate_square_covariance_matrix(cov, "MeanVarianceOptimizer")
        cov = resolve_selected_covariance(cov, selected)
        rets = rets.reindex(selected)

        asset_count = len(selected)
        if asset_count == 0:
            self.set_problem_data(asset_count=0, universe=[], weights_var=None)
            self._constraints = []
            self._objective = lambda: cvx.Maximize(0)
            return

        cov_matrix = build_covariance_matrix(cov)
        exp_returns = build_expected_returns_parameter(
            rets,
            self.parameter,
        )
        weights = build_weight_variable(
            asset_count,
            self.variable,
        )
        min_weights, max_weights = self.compute_weight_bounds(selected, bounds)

        self.add_objective(
            lambda: mean_variance_utility_objective(
                weights,
                exp_returns,
                cov_matrix,
                float(risk_averse_lambda),
                self.maximize,
            )
        )
        self.bulk_add_constraints(
            bound_constraints(
                weights,
                min_weights,
                max_weights,
            )
        )
        self.add_constraint(sum_to_one_constraint(weights))
        self.set_problem_data(
            universe=selected,
            asset_count=asset_count,
            weights_var=weights,
            min_weights=min_weights,
            max_weights=max_weights,
        )

    def solve_problem(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Solve convex problem and map solution vector to asset dictionary."""
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
            raise RuntimeError("MeanVarianceOptimizer solved without weights.")
        solved = np.asarray(weights_var.value, dtype=float).reshape(-1)
        self.weights_ = {
            asset: float(weight) for asset, weight in zip(universe, solved)
        }
        return self.get_result()


class WeightMeanVar(WeightAlgo):
    """Assign portfolio weights using convex mean-variance optimization."""

    def __init__(
        self,
        risk_averse_lambda: float = 1.0,
        bounds: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.risk_averse_lambda = float(risk_averse_lambda)
        self.bounds = bounds
        self.optimizer = MeanVarianceOptimizer()

    def __call__(self, target: Any) -> bool:
        """Compute and store mean-variance weights in ``temp['weights']``."""
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

        rets = temp.get("expected_returns")
        cov = temp.get("covariance")
        self.optimizer.set_problem(
            rets,
            cov,
            selected=selected_raw,
            risk_averse_lambda=self.risk_averse_lambda,
            bounds=self.bounds,
        )
        result = self.optimizer.solve_problem()
        weights = result["weights"]
        self._write_weights(temp, weights)
        self._record_allocation_history(now, weights)
        return True
