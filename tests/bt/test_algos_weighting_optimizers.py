import pytest

from bt.algos.weighting.optimizers.base_optimizer import BaseOptimizer
from bt.algos.weighting.optimizers.convex_optimizer import ConvexOptimizer
import bt.algos.weighting.optimizers.convex_optimizer as convex_mod


class DummyOptimizer(BaseOptimizer):
    def solve(self, *args, **kwargs):
        self.validate_problem()
        self.success = True
        self.status = "optimal"
        self.message = "ok"
        self.weights_ = {"c1": 1.0}
        return self.get_result()


class HookOptimizer(BaseOptimizer):
    def __init__(self):
        super().__init__()
        self.hook_called = False

    def check_inputs(self) -> None:
        self.hook_called = True


def test_base_optimizer_initial_state():
    opt = DummyOptimizer()
    assert opt.objective is None
    assert opt.constraints == []
    assert opt.problem_data == {}
    assert opt.solver_options == {}
    assert opt.weights_ is None
    assert opt.success is False
    assert opt.status is None
    assert opt.message is None


def test_add_objective_validates_callable():
    opt = DummyOptimizer()
    with pytest.raises(TypeError, match="must be callable"):
        opt.add_objective(123)  # type: ignore[arg-type]

    fn = lambda x: x
    opt.add_objective(fn)
    assert opt.objective is fn


def test_add_constraint_and_constraints_property_copy():
    opt = DummyOptimizer()
    c = object()
    opt.add_constraint(c)
    assert opt.constraints == [c]

    snapshot = opt.constraints
    snapshot.append("new")
    assert opt.constraints == [c]


def test_set_problem_data_and_solver_options():
    opt = DummyOptimizer()
    opt.set_problem_data(returns="r", cov="c")
    opt.set_solver_options(solver="ECOS", max_iters=1000)

    assert opt.problem_data == {"returns": "r", "cov": "c"}
    assert opt.solver_options == {"solver": "ECOS", "max_iters": 1000}


def test_validate_problem_requires_objective():
    opt = DummyOptimizer()
    with pytest.raises(ValueError, match="requires an objective"):
        opt.validate_problem()


def test_validate_problem_calls_check_inputs_hook():
    opt = HookOptimizer()
    opt.add_objective(lambda: None)
    opt.validate_problem()
    assert opt.hook_called is True


def test_get_result_returns_standard_payload():
    opt = DummyOptimizer()
    opt.weights_ = {"c1": 0.7, "c2": 0.3}
    opt.success = True
    opt.status = "optimal"
    opt.message = "done"

    result = opt.get_result()
    assert result == {
        "weights": {"c1": 0.7, "c2": 0.3},
        "success": True,
        "status": "optimal",
        "message": "done",
    }


def test_base_solve_raises_not_implemented():
    class NoSolveOptimizer(BaseOptimizer):
        pass

    opt = NoSolveOptimizer()
    opt.add_objective(lambda: None)
    with pytest.raises(NotImplementedError, match="Subclasses must implement"):
        opt.solve()


def test_reset_clears_runtime_state():
    opt = DummyOptimizer()
    opt.add_objective(lambda: None)
    opt.add_constraint("c")
    opt.set_problem_data(returns="r")
    opt.set_solver_options(solver="SCS")
    opt.weights_ = {"c1": 1.0}
    opt.success = True
    opt.status = "optimal"
    opt.message = "ok"

    opt.reset()

    assert opt.objective is None
    assert opt.constraints == []
    assert opt.problem_data == {}
    assert opt.solver_options == {}
    assert opt.weights_ is None
    assert opt.success is False
    assert opt.status is None
    assert opt.message is None


def test_dummy_optimizer_solve_returns_result_payload():
    opt = DummyOptimizer()
    opt.add_objective(lambda: None)
    result = opt.solve()

    assert result["success"] is True
    assert result["status"] == "optimal"
    assert result["message"] == "ok"
    assert result["weights"] == {"c1": 1.0}


def test_convex_optimizer_requires_callable_objective():
    opt = ConvexOptimizer()
    # bypass BaseOptimizer.add_objective to test Convex-specific validation
    opt._objective = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="objective must be a callable"):
        opt.solve()


def test_convex_optimizer_requires_solver_candidates():
    opt = ConvexOptimizer(solver_candidates=())
    opt.add_objective(lambda: convex_mod.cvx.Minimize(0))
    with pytest.raises(ValueError, match="at least one solver candidate"):
        opt.solve()


def test_convex_optimizer_builder_none_raises():
    opt = ConvexOptimizer()
    opt.add_objective(lambda: None)
    with pytest.raises(ValueError, match="returned None"):
        opt.solve()


def test_convex_optimizer_solves_simple_problem():
    x = convex_mod.cvx.Variable()
    opt = ConvexOptimizer()
    opt.add_objective(lambda: convex_mod.cvx.Minimize(convex_mod.cvx.square(x - 1.0)))
    opt.add_constraint(lambda: x >= 0.0)
    result = opt.solve()

    assert result["success"] is True
    assert result["status"] in {"optimal", "optimal_inaccurate"}
    assert isinstance(result["message"], str)
