from __future__ import annotations

from abc import ABC
from typing import Any, Callable


class BaseOptimizer(ABC):
    """Base contract and shared utilities for portfolio optimizers.

    This class stores optimizer configuration/state and defines a single
    execution entrypoint: ``solve_problem(...)``.
    """

    MAX_ITER = 300

    def __init__(self) -> None:
        self._objective: Callable[..., Any] | None = None
        self._constraints: list[Any] = []
        self.problem_data: dict[str, Any] = {}
        self.solver_options: dict[str, Any] = {}
        self.weights_: Any = None
        self.success = False
        self.status: str | None = None
        self.message: str | None = None

    @property
    def objective(self) -> Callable[..., Any] | None:
        """Return currently configured objective callable."""
        return self._objective

    @property
    def constraints(self) -> list[Any]:
        """Return current constraint list (read-only snapshot semantics)."""
        return list(self._constraints)

    def reset(self) -> None:
        """Clear objective/state and reset result metadata."""
        self._objective = None
        self._constraints = []
        self.problem_data = {}
        self.solver_options = {}
        self.weights_ = None
        self.success = False
        self.status = None
        self.message = None

    def add_objective(self, obj: Callable[..., Any]) -> None:
        """Set objective callable used by the optimizer.

        Parameters
        ----------
        obj : callable
            Objective function or backend-specific expression builder.
        """
        if not callable(obj):
            raise TypeError("BaseOptimizer `obj` must be callable.")
        self._objective = obj

    def add_constraint(self, new_constraint: Any) -> None:
        """Append one constraint to the optimizer state."""
        self._constraints.append(new_constraint)

    def set_problem_data(self, **problem_data: Any) -> None:
        """Set optimization input data used by concrete solvers.

        Parameters
        ----------
        **problem_data
            Arbitrary key/value payload (for example ``returns``, ``cov``,
            ``bounds``) consumed by subclass implementations.
        """
        self.problem_data = dict(problem_data)

    def set_solver_options(self, **solver_options: Any) -> None:
        """Set backend solver options used by subclass implementations."""
        self.solver_options = dict(solver_options)

    def check_inputs(self) -> None:
        """Validate input payload before solving.

        Subclasses should override this hook with backend/problem-specific
        validation (shape checks, PSD checks, bounds checks, etc.).
        """
        return None

    def validate_problem(self) -> None:
        """Validate that objective and constraints are ready for solving."""
        if self._objective is None:
            raise ValueError("BaseOptimizer requires an objective before solving.")
        self.check_inputs()

    def get_result(self) -> dict[str, Any]:
        """Return standardized optimization result metadata."""
        return {
            "weights": self.weights_,
            "success": self.success,
            "status": self.status,
            "message": self.message,
        }

    def solve_problem(self, *args: Any, **kwargs: Any) -> Any:
        """Solve the optimization problem.

        Concrete subclasses should override this method and return a payload
        compatible with ``get_result()`` semantics where applicable.
        """
        raise NotImplementedError("Subclasses must implement `solve_problem`.")
