import pytest
import pandas as pd
import numpy as np

from bt.algos.weighting.optimizers.base_optimizer import BaseOptimizer
from bt.algos.weighting.optimizers.convex_optimizer import ConvexOptimizer
from bt.algos.weighting.optimizers.constraints import (
    bound_constraints,
    sum_to_one_constraint,
)
from bt.algos.weighting.optimizers.objectives import negative_sharpe_ratio
from bt.algos.weighting.optimizers.objectives import mean_variance_utility_objective
from bt.algos.weighting.optimizers.validators import (
    resolve_selected_covariance,
    validate_series,
    validate_square_covariance_matrix,
)
from bt.algos.weighting.mean_variance import MeanVarianceOptimizer
import bt.algos.weighting.optimizers.convex_optimizer as convex_mod
import cvxpy as cvx


class DummyOptimizer(BaseOptimizer):
    def solve_problem(self, *args, **kwargs):
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


def test_bulk_add_constraints_appends_all():
    opt = DummyOptimizer()
    c1 = object()
    c2 = object()
    opt.bulk_add_constraints([c1, c2])
    assert opt.constraints == [c1, c2]


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


def test_base_solve_problem_raises_not_implemented():
    class NoSolveProblemOptimizer(BaseOptimizer):
        pass

    opt = NoSolveProblemOptimizer()
    opt.add_objective(lambda: None)
    with pytest.raises(NotImplementedError, match="Subclasses must implement"):
        opt.solve_problem()


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


def test_dummy_optimizer_solve_problem_returns_result_payload():
    opt = DummyOptimizer()
    opt.add_objective(lambda: None)
    result = opt.solve_problem()

    assert result["success"] is True
    assert result["status"] == "optimal"
    assert result["message"] == "ok"
    assert result["weights"] == {"c1": 1.0}


def test_convex_optimizer_requires_callable_objective():
    opt = ConvexOptimizer()
    # bypass BaseOptimizer.add_objective to test Convex-specific validation
    opt._objective = object()  # type: ignore[assignment]
    with pytest.raises(TypeError, match="objective must be a callable"):
        opt.solve_problem()


def test_convex_optimizer_requires_solver_candidates():
    opt = ConvexOptimizer(solver_candidates=())
    opt.add_objective(lambda: convex_mod.cvx.Minimize(0))
    with pytest.raises(ValueError, match="at least one solver candidate"):
        opt.solve_problem()


def test_convex_optimizer_builder_none_raises():
    opt = ConvexOptimizer()
    opt.add_objective(lambda: None)
    with pytest.raises(ValueError, match="returned None"):
        opt.solve_problem()


def test_convex_optimizer_solves_simple_problem():
    x = convex_mod.cvx.Variable()
    opt = ConvexOptimizer()
    opt.add_objective(lambda: convex_mod.cvx.Minimize(convex_mod.cvx.square(x - 1.0)))
    opt.add_constraint(lambda: x >= 0.0)
    result = opt.solve_problem()

    assert result["success"] is True
    assert result["status"] in {"optimal", "optimal_inaccurate"}
    assert isinstance(result["message"], str)


def test_validate_square_covariance_matrix_rejects_non_dataframe():
    with pytest.raises(TypeError, match="must be a DataFrame"):
        validate_square_covariance_matrix([1, 2, 3], "TestOpt")  # type: ignore[arg-type]


def test_validate_square_covariance_matrix_rejects_non_square():
    cov = pd.DataFrame([[1.0, 0.1]], index=["a"], columns=["a", "b"])
    with pytest.raises(ValueError, match="must be square"):
        validate_square_covariance_matrix(cov, "TestOpt")


def test_validate_square_covariance_matrix_rejects_axis_mismatch():
    cov = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], index=["a", "b"], columns=["a", "c"])
    with pytest.raises(ValueError, match="index/columns must match"):
        validate_square_covariance_matrix(cov, "TestOpt")


def test_validate_square_covariance_matrix_accepts_valid_input():
    cov = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], index=["a", "b"], columns=["a", "b"])
    validate_square_covariance_matrix(cov, "TestOpt")


def test_validate_series_rejects_non_series():
    with pytest.raises(TypeError, match="must be a Series"):
        validate_series({"a": 1.0}, "TestOpt", "market_caps")


def test_validate_series_accepts_series():
    validate_series(pd.Series({"a": 1.0}), "TestOpt", "market_caps")


def test_resolve_selected_covariance_subset_failure_returns_empty():
    cov = pd.DataFrame([[1.0]], index=["a"], columns=["a"])
    subset = resolve_selected_covariance(cov, ["b"])
    assert subset.empty


def test_resolve_selected_covariance_success():
    cov = pd.DataFrame(
        [[1.0, 0.2], [0.2, 1.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    subset = resolve_selected_covariance(cov, ["b", "a"])
    assert list(subset.index) == ["b", "a"]
    assert list(subset.columns) == ["b", "a"]


def test_sum_to_one_constraint():
    w = cvx.Variable(2)
    constraint = sum_to_one_constraint(w)
    problem = cvx.Problem(cvx.Minimize(0), [constraint, w >= 0])
    problem.solve()
    assert w.value is not None
    assert np.sum(w.value) == pytest.approx(1.0)


def test_negative_sharpe_ratio():
    w = np.array([0.5, 0.5])
    mu = np.array([0.1, 0.2])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    value = negative_sharpe_ratio(w, mu, cov, rf=0.0)
    assert np.isfinite(value)
    assert value < 0.0


def test_mean_variance_utility_objective_builder():
    w = cvx.Variable(2)
    mu = cvx.Parameter(2)
    mu.value = np.array([0.1, 0.2])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    obj = mean_variance_utility_objective(
        w,
        mu,
        cov,
        risk_averse_lambda=1.0,
        maximize_builder=cvx.Maximize,
    )
    assert isinstance(obj, cvx.Maximize)


def test_mean_variance_constraints_builders():
    w = cvx.Variable(2)
    box = bound_constraints(
        w,
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
    )
    assert len(box) == 2


def test_mean_variance_optimizer_set_and_solve():
    expected_returns = pd.Series({"c1": 0.01, "c2": 0.02})
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]],
        index=["c1", "c2"],
        columns=["c1", "c2"],
    )
    opt = MeanVarianceOptimizer()
    opt.set_problem(expected_returns, covariance, ["c1", "c2"])
    result = opt.solve_problem()
    assert result["success"] is True
    assert set(result["weights"].keys()) == {"c1", "c2"}
    assert sum(result["weights"].values()) == pytest.approx(1.0)


def test_mean_variance_optimizer_set_problem_data_contract():
    expected_returns = pd.Series({"c1": 0.01, "c2": 0.02, "c3": 0.03})
    covariance = pd.DataFrame(
        [[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.16]],
        index=["c1", "c2", "c3"],
        columns=["c1", "c2", "c3"],
    )
    selected = ["c3", "c1"]

    opt = MeanVarianceOptimizer()
    opt.set_problem(
        expected_returns,
        covariance,
        selected=selected,
        bounds=(0.1, 0.8),
    )

    assert opt.problem_data["universe"] == selected
    assert opt.problem_data["asset_count"] == 2
    assert opt.problem_data["weights_var"] is not None
    np.testing.assert_allclose(opt.problem_data["min_weights"], np.array([0.1, 0.1]))
    np.testing.assert_allclose(opt.problem_data["max_weights"], np.array([0.8, 0.8]))
    assert len(opt.constraints) == 3
    result = opt.solve_problem()
    assert result["success"] is True
    weights = result["weights"]
    assert set(weights.keys()) == set(selected)
    total = sum(weights.values())
    assert total == pytest.approx(1.0)
    for weight in weights.values():
        assert 0.1 <= weight <= 0.8


def test_mean_variance_optimizer_respects_selected_order():
    expected_returns = pd.Series({"c1": 0.01, "c2": 0.03})
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["c1", "c2"],
        columns=["c1", "c2"],
    )
    selected = ["c2", "c1"]

    opt = MeanVarianceOptimizer()
    opt.set_problem(expected_returns, covariance, selected=selected)
    result = opt.solve_problem()

    assert list(result["weights"].keys()) == selected
    assert sum(result["weights"].values()) == pytest.approx(1.0)


def test_mean_variance_optimizer_single_asset():
    expected_returns = pd.Series({"c1": 0.01})
    covariance = pd.DataFrame([[0.04]], index=["c1"], columns=["c1"])
    opt = MeanVarianceOptimizer()
    opt.set_problem(expected_returns, covariance, ["c1"])
    result = opt.solve_problem()
    assert result["weights"] == {"c1": pytest.approx(1.0)}


def test_mean_variance_optimizer_empty_returns():
    expected_returns = pd.Series(dtype=float)
    covariance = pd.DataFrame()
    opt = MeanVarianceOptimizer()
    opt.set_problem(expected_returns, covariance, [])
    result = opt.solve_problem()
    assert result["weights"] == {}
