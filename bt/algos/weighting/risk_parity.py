from __future__ import annotations

from typing import Any

import cvxpy as cvx
import numpy as np
import pandas as pd

from bt.algos.weighting.core import WeightAlgo
from bt.algos.weighting.optimizers.constraints import (
    non_negative_constraint,
    sum_to_one_constraint,
)
from bt.algos.weighting.optimizers.convex_optimizer import ConvexOptimizer
from bt.algos.weighting.optimizers.objectives import risk_parity_objective
from bt.algos.weighting.optimizers.validators import (
    resolve_selected_covariance,
    validate_square_covariance_matrix,
)
from bt.algos.weighting.optimizers.variables import (
    build_covariance_matrix,
    build_weight_variable,
)


class RiskParityOptimizer(ConvexOptimizer):
    """Convex equal-risk-budget (risk parity) optimizer.

    Optimizes:
    ``min_w (0.5 * w^T Sigma w - b^T log(w))``
    with equal budgets ``b`` and constraints ``w >= 0``, ``sum(w)=1``.
    """

    def __init__(self) -> None:
        super().__init__()

    def set_problem(
        self,
        cov: pd.DataFrame,
        selected: list[str],
        **kwargs: Any,
    ) -> None:
        """Build risk-parity CVXPY problem for selected assets."""
        self.reset()
        validate_square_covariance_matrix(cov, "RiskParityOptimizer")
        cov = resolve_selected_covariance(cov, selected)

        asset_count = len(selected)
        if asset_count == 0:
            self.set_problem_data(asset_count=0, universe=[], weights_var=None)
            self._constraints = []
            self._objective = lambda: cvx.Maximize(0)
            return

        cov_matrix = build_covariance_matrix(cov)
        weights = build_weight_variable(asset_count, self.variable)
        risk_budgets = np.full(asset_count, 1.0 / asset_count, dtype=float)

        self.add_objective(
            lambda: risk_parity_objective(
                weights,
                cov_matrix,
                risk_budgets,
                self.minimize,
            )
        )
        self.add_constraint(non_negative_constraint(weights))
        self.add_constraint(sum_to_one_constraint(weights))
        self.set_problem_data(
            universe=selected,
            asset_count=asset_count,
            weights_var=weights,
        )

    def solve_problem(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Solve risk-parity optimization and return weights payload."""
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
            raise RuntimeError("RiskParityOptimizer solved without weights.")
        solved = np.asarray(weights_var.value, dtype=float).reshape(-1)
        self.weights_ = {
            asset: float(weight) for asset, weight in zip(universe, solved)
        }
        return self.get_result()


class WeightRiskParity(WeightAlgo):
    """Assign ERC/risk-parity weights from strategy ``temp`` state.

    Inputs expected in ``target.temp``:
    - ``selected``: names to allocate
    - covariance at ``self.covariance_key`` (default ``covariance``)
    """

    def __init__(
        self,
        covariance_key: str = "covariance",
    ) -> None:
        super().__init__()
        self.covariance_key = covariance_key
        self.optimizer = RiskParityOptimizer()

    def __call__(self, target: Any) -> bool:
        """Compute and store risk-parity weights."""
        temp = self._resolve_temp(target)
        if temp is None:
            return False
        now = self._resolve_now(target)

        selected_raw = temp.get("selected", [])
        if not isinstance(selected_raw, list):
            return False
        if not selected_raw:
            self._write_weights(temp, {}, now=now, record_history=True)
            return True
        if len(selected_raw) == 1:
            weights = {selected_raw[0]: 1.0}
            self._write_weights(temp, weights, now=now, record_history=True)
            return True

        covariance = temp.get(self.covariance_key)
        self.optimizer.set_problem(covariance, selected_raw)
        result = self.optimizer.solve_problem()
        weights = result["weights"]
        self._write_weights(temp, weights, now=now, record_history=True)
        return True
