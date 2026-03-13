from typing import Any

import numpy as np
import pandas as pd

from bt.algos.weighting.core import WeightAlgo
from bt.algos.weighting.optimizers.base_optimizer import BaseOptimizer
from bt.algos.weighting.optimizers.validators import (
    resolve_selected_covariance,
    validate_square_covariance_matrix,
)


class InvVolOptimizer(BaseOptimizer):
    """Analytical inverse-volatility optimizer.

    For valid covariance input, this optimizer sets:
    ``w_i ∝ 1 / sqrt(cov_ii)``.
    """

    def __init__(self) -> None:
        super().__init__()

    def set_problem(self, covariance: Any, selected: list[str] | None = None) -> None:
        """Validate and store covariance input for optimization.

        Parameters
        ----------
        covariance
            Covariance matrix of asset returns.
        selected
            Optional selected universe used to subset covariance.
        """
        self.reset()
        validate_square_covariance_matrix(covariance, "InvVolOptimizer")
        if selected is not None:
            covariance = resolve_selected_covariance(covariance, selected)
        self.set_problem_data(covariance=covariance, assets=list(covariance.columns))

    def solve_problem(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Compute inverse-volatility weights and return result metadata."""
        covariance = self.problem_data.get("covariance")
        assets = self.problem_data.get("assets", [])

        if covariance is None or covariance.empty:
            allocations: dict[str, float] = {}
        else:
            diagonal = np.diag(covariance.to_numpy(dtype=float, copy=True))
            vol = np.sqrt(diagonal)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_vol = np.where(np.isfinite(vol) & (vol > 0.0), 1.0 / vol, np.nan)

            inv_vol_sum = np.nansum(inv_vol)
            if not np.isfinite(inv_vol_sum) or inv_vol_sum <= 0.0:
                allocations = {}
            else:
                weights = inv_vol / inv_vol_sum
                allocations = {
                    asset: float(weight)
                    for asset, weight in zip(assets, weights)
                    if np.isfinite(weight)
                }

        self.weights_ = allocations
        self.success = True
        self.status = "optimal"
        self.message = "Solved analytically."
        return self.get_result()


class WeightInvVol(WeightAlgo):
    """Assign inverse-volatility weights from strategy ``temp`` state.

    Inputs expected in ``target.temp``:
    - ``selected``: list of names to allocate.
    - ``covariance``: covariance matrix aligned to candidate names.
    """

    def __init__(self) -> None:
        """Initialize inverse-volatility assigner."""
        super().__init__()
        self.optimizer = InvVolOptimizer()

    def __call__(self, target: Any) -> bool:
        """Compute and store inverse-volatility weights.

        Returns
        -------
        bool
            ``True`` when processed, ``False`` for invalid context/state.
        """
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

        covariance = temp.get("covariance")
        self.optimizer.set_problem(covariance, selected_raw)
        result = self.optimizer.solve_problem()
        weights = result["weights"]
        self._write_weights(temp, weights, now=now, record_history=True)
        return True
