from __future__ import annotations

from typing import Any, Callable

import numpy as np
import cvxpy as cvx

from .base_optimizer import BaseOptimizer


class ConvexOptimizer(BaseOptimizer):
    """CVXPY-backed convex optimization engine.

    This optimizer expects objective/constraint builders to be configured on the
    base state and solves using the first available solver from
    ``solver_candidates``.

    Objective contract
    ------------------
    ``add_objective(builder)`` should receive a callable that returns a CVXPY
    objective object (for example ``cvx.Minimize(expr)``).

    Constraint contract
    -------------------
    ``add_constraint(item)`` accepts either:
    - a ready CVXPY constraint expression, or
    - a callable returning a CVXPY constraint expression.
    """

    DEFAULT_SOLVER_CANDIDATES = ("CLARABEL", "ECOS", "SCS", "OSQP", "CVXOPT")

    def __init__(self, solver_candidates: tuple[str, ...] | None = None) -> None:
        """Initialize CVXPY optimizer state."""
        super().__init__()
        self.problem: Any = None
        self.solver_candidates = (
            self.DEFAULT_SOLVER_CANDIDATES
            if solver_candidates is None
            else solver_candidates
        )

    @staticmethod
    def variable(shape=(), **kwargs: Any) -> Any:
        """Create a CVXPY variable."""
        return cvx.Variable(shape=shape, **kwargs)

    @staticmethod
    def parameter(shape=(), **kwargs: Any) -> Any:
        """Create a CVXPY parameter."""
        return cvx.Parameter(shape=shape, **kwargs)

    @staticmethod
    def minimize(expr: Any) -> Any:
        """Build a CVXPY minimize objective."""
        return cvx.Minimize(expr)

    @staticmethod
    def maximize(expr: Any) -> Any:
        """Build a CVXPY maximize objective."""
        return cvx.Maximize(expr)

    @staticmethod
    def compute_weight_bounds(
        assets: list[str],
        bounds: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build per-asset min/max bound arrays from one ``bounds`` tuple."""
        default_min, default_max = bounds
        min_weights = np.full(len(assets), float(default_min), dtype=float)
        max_weights = np.full(len(assets), float(default_max), dtype=float)
        return min_weights, max_weights

    def check_inputs(self) -> None:
        """Validate convex optimizer setup before solving."""
        if not callable(self.objective):
            raise TypeError(
                "ConvexOptimizer objective must be a callable returning a CVXPY objective."
            )
        if not self.solver_candidates:
            raise ValueError("ConvexOptimizer requires at least one solver candidate.")

    def _resolve_objective(self) -> Any:
        """Build concrete CVXPY objective from configured builder."""
        objective_builder = self.objective
        if objective_builder is None:
            raise ValueError("ConvexOptimizer requires an objective builder.")
        objective = objective_builder()
        if objective is None:
            raise ValueError("ConvexOptimizer objective builder returned None.")
        return objective

    def _resolve_constraints(self) -> list[Any]:
        """Build concrete CVXPY constraints from configured constraints."""
        resolved: list[Any] = []
        for item in self.constraints:
            constraint = item() if callable(item) else item
            if constraint is None:
                continue
            resolved.append(constraint)
        return resolved

    def solve_problem(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Form and solve the CVXPY problem.

        Returns
        -------
        dict[str, Any]
            Standard result payload from ``BaseOptimizer.get_result``.
        """
        self.validate_problem()

        objective = self._resolve_objective()
        constraints = self._resolve_constraints()
        self.problem = cvx.Problem(objective, constraints)

        self.success = False
        self.status = None
        self.message = None

        last_error: Exception | None = None
        for solver_name in self.solver_candidates:
            try:
                self.problem.solve(solver=solver_name, **self.solver_options)
                self.status = str(self.problem.status)
                if self.problem.status in {"optimal", "optimal_inaccurate"}:
                    self.success = True
                    self.message = f"Solved with {solver_name}."
                    break
                self.message = (
                    f"Solver {solver_name} returned status {self.problem.status}."
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self.message = f"Solver {solver_name} failed: {exc}"

        if not self.success and last_error is not None:
            raise RuntimeError(str(last_error)) from last_error
        if not self.success:
            raise RuntimeError(self.message or "Convex optimization failed.")

        return self.get_result()
