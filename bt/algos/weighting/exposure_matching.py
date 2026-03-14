from typing import Any

import cvxpy as cvx
import numpy as np
import pandas as pd

from bt.algos.weighting.core import WeightAlgo
from bt.algos.weighting.optimizers.constraints import (
    bound_constraints,
    non_negative_constraint,
    sum_to_zero_constraint,
)
from bt.algos.weighting.optimizers.convex_optimizer import ConvexOptimizer
from bt.algos.weighting.optimizers.objectives import exposure_matching_objective
from bt.algos.weighting.optimizers.validators import (
    validate_dataframe,
    validate_series,
    validate_square_covariance_matrix,
)
from bt.algos.weighting.optimizers.variables import (
    build_covariance_matrix,
    build_matrix_parameter,
    build_series_parameter,
    build_weight_variable,
)
from utils.dataframe_utils import coerce_numeric_frame, coerce_numeric_series
from utils.math_utils import validate_non_negative, validate_real


class ExposureMatchingOptimizer(ConvexOptimizer):
    """Convex active portfolio optimizer with factor-risk penalties.

    The optimizer solves for active weights ``x`` relative to a benchmark:

    ``max_x x^T s - lambda_factor * (B^T x)^T Sigma_f (B^T x)``

    subject to:
    - ``sum(x) = 0``
    - optional active-weight box bounds
    - long-only total weights on ``w = w_bench + x``
    """

    def __init__(
        self,
        lambda_factor: float = 1.0,
        active_bound: float | None = None,
    ) -> None:
        super().__init__()
        self.lambda_factor = validate_non_negative(
            validate_real(lambda_factor, "ExposureMatchingOptimizer `lambda_factor`"),
            "ExposureMatchingOptimizer `lambda_factor`",
        )
        self.active_bound = (
            None
            if active_bound is None
            else validate_non_negative(
                validate_real(active_bound, "ExposureMatchingOptimizer `active_bound`"),
                "ExposureMatchingOptimizer `active_bound`",
            )
        )
        self.active_weights_: dict[str, float] = {}

    def reset(self) -> None:
        """Reset optimizer state and active-weight solution cache."""
        super().reset()
        self.active_weights_ = {}

    def _align_problem_inputs(
        self,
        stat: pd.Series,
        benchmark_weights: pd.Series,
        factor_exposures: pd.DataFrame,
        factor_covariance: pd.DataFrame,
        selected: list[str],
    ) -> tuple[list[str], pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        """Validate and align all problem inputs to ``selected`` order."""
        label = "ExposureMatchingOptimizer"
        validate_series(stat, label, "stat")
        validate_series(benchmark_weights, label, "benchmark_weights")
        validate_dataframe(factor_exposures, label, "factor_exposures")
        validate_square_covariance_matrix(factor_covariance, label)

        selected_index = pd.Index(selected)
        available_selected = selected_index[
            selected_index.isin(stat.index)
            & selected_index.isin(benchmark_weights.index)
            & selected_index.isin(factor_exposures.index)
        ]
        surviving_selected = available_selected.tolist()

        aligned_stat = coerce_numeric_series(
            stat.reindex(surviving_selected),
            label,
            "stat",
        )
        aligned_benchmark = coerce_numeric_series(
            benchmark_weights.reindex(surviving_selected),
            label,
            "benchmark_weights",
        )
        aligned_exposures = coerce_numeric_frame(
            factor_exposures.reindex(index=surviving_selected),
            label,
            "factor_exposures",
        )

        if not aligned_benchmark.empty:
            benchmark_total = float(aligned_benchmark.sum())
            if not np.isfinite(benchmark_total) or np.isclose(benchmark_total, 0.0):
                raise ValueError(
                    f"{label} surviving `benchmark_weights` must sum to a non-zero value."
                )
            aligned_benchmark = aligned_benchmark / benchmark_total

        factor_names = pd.Index(aligned_exposures.columns)
        covariance_factors = pd.Index(factor_covariance.index)
        if not factor_names.equals(covariance_factors):
            missing_factor_cov = factor_names.difference(covariance_factors)
            extra_factor_cov = covariance_factors.difference(factor_names)
            raise ValueError(
                f"{label} `factor_exposures` columns and `factor_covariance` columns must match. "
                f"Missing in covariance: {list(missing_factor_cov)}. "
                f"Extra in covariance: {list(extra_factor_cov)}."
            )
        aligned_factor_covariance = coerce_numeric_frame(
            factor_covariance.loc[factor_names.tolist(), factor_names.tolist()],
            label,
            "factor_covariance",
        )
        return (
            surviving_selected,
            aligned_stat,
            aligned_benchmark,
            aligned_exposures,
            aligned_factor_covariance,
        )

    def set_problem(
        self,
        stat: pd.Series,
        benchmark_weights: pd.Series,
        factor_exposures: pd.DataFrame,
        factor_covariance: pd.DataFrame,
        selected: list[str],
        **kwargs: Any,
    ) -> None:
        """Build the convex exposure-matching problem for one cross section."""
        self.reset()

        (
            surviving_selected,
            aligned_stat,
            aligned_benchmark,
            aligned_exposures,
            aligned_factor_covariance,
        ) = self._align_problem_inputs(
            stat,
            benchmark_weights,
            factor_exposures,
            factor_covariance,
            selected,
        )

        asset_count = len(surviving_selected)
        if asset_count == 0:
            self.set_problem_data(
                asset_count=0,
                universe=[],
                active_weights_var=None,
                benchmark_weights=np.array([], dtype=float),
            )
            self._constraints = []
            self._objective = lambda: cvx.Maximize(0)
            return

        if asset_count == 1:
            self.set_problem_data(
                asset_count=1,
                universe=surviving_selected,
                active_weights_var=None,
                benchmark_weights=aligned_benchmark.to_numpy(dtype=float, copy=True),
            )
            self._constraints = []
            self._objective = lambda: cvx.Maximize(0)
            return

        signal_stats = build_series_parameter(
            aligned_stat,
            self.parameter,
        )
        benchmark_vector = build_series_parameter(
            aligned_benchmark,
            self.parameter,
        )
        active_weights = build_weight_variable(
            asset_count,
            self.variable,
        )
        total_weights = benchmark_vector + active_weights

        factor_exposure_matrix = build_matrix_parameter(
            aligned_exposures,
            self.parameter,
        )
        factor_covariance_matrix = build_covariance_matrix(aligned_factor_covariance)

        self.add_objective(
            lambda: exposure_matching_objective(
                active_weights,
                signal_stats,
                factor_exposure_matrix,
                factor_covariance_matrix,
                self.lambda_factor,
                self.maximize,
            )
        )
        self.add_constraint(sum_to_zero_constraint(active_weights))
        if self.active_bound is not None:
            min_active, max_active = self.compute_weight_bounds(
                surviving_selected,
                (-self.active_bound, self.active_bound),
            )
            self.bulk_add_constraints(
                bound_constraints(active_weights, min_active, max_active)
            )
        self.add_constraint(non_negative_constraint(total_weights))

        self.set_problem_data(
            universe=surviving_selected,
            asset_count=asset_count,
            active_weights_var=active_weights,
            benchmark_weights=aligned_benchmark.to_numpy(dtype=float, copy=True),
        )

    def get_result(self) -> dict[str, Any]:
        """Return standard optimizer result plus active-weight payload."""
        result = super().get_result()
        result["active_weights"] = self.active_weights_
        return result

    def solve_problem(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Solve exposure-matching optimization and return total/active weights."""
        asset_count = int(self.problem_data.get("asset_count", 0))
        universe = self.problem_data.get("universe", [])
        active_weights_var = self.problem_data.get("active_weights_var")
        benchmark_weights = np.asarray(
            self.problem_data.get("benchmark_weights", np.array([], dtype=float)),
            dtype=float,
        ).reshape(-1)

        if asset_count == 0:
            self.weights_ = {}
            self.active_weights_ = {}
            self.success = True
            self.status = "optimal"
            self.message = "No assets to optimize."
            return self.get_result()
        if asset_count == 1:
            benchmark_weight = (
                float(benchmark_weights[0]) if benchmark_weights.size else 0.0
            )
            self.weights_ = {universe[0]: benchmark_weight}
            self.active_weights_ = {universe[0]: 0.0}
            self.success = True
            self.status = "optimal"
            self.message = "Single asset."
            return self.get_result()

        super().solve_problem(*args, **kwargs)
        if active_weights_var is None or active_weights_var.value is None:
            raise RuntimeError(
                "ExposureMatchingOptimizer solved without active weights."
            )

        solved_active = np.asarray(active_weights_var.value, dtype=float).reshape(-1)
        solved_total = benchmark_weights + solved_active
        self.active_weights_ = {
            asset: float(weight) for asset, weight in zip(universe, solved_active)
        }
        self.weights_ = {
            asset: float(weight) for asset, weight in zip(universe, solved_total)
        }
        return self.get_result()


class ExposureMatching(WeightAlgo):
    """Assign weights via benchmark-relative exposure-matching optimization.

    Inputs expected in ``target.temp``:
    - ``selected``: names to allocate
    - preferably standardized stat series at ``stat_key``
    - benchmark weight series at ``temp["benchmark_weights"]``
    - stock-level factor exposures DataFrame at ``temp["factor_exposures"]``
    - factor covariance matrix at ``temp["factor_covariance"]``
    """

    def __init__(
        self,
        stat_key: str,
        lambda_factor: float = 1.0,
        active_bound: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(stat_key, str) or not stat_key:
            raise TypeError("ExposureMatching `stat_key` must be a non-empty string.")

        self.stat_key = stat_key
        self.optimizer = ExposureMatchingOptimizer(
            lambda_factor=lambda_factor,
            active_bound=active_bound,
        )

    def __call__(self, target: Any) -> bool:
        """Run exposure-matching optimization and write ``temp['weights']``."""
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

        self.optimizer.set_problem(
            stat=temp.get(self.stat_key),
            benchmark_weights=temp.get("benchmark_weights"),
            factor_exposures=temp.get("factor_exposures"),
            factor_covariance=temp.get("factor_covariance"),
            selected=selected_raw,
        )
        result = self.optimizer.solve_problem()
        self._write_weights(temp, result["weights"], now=now, record_history=True)
        return True
