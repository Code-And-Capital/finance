from types import SimpleNamespace

import pandas as pd

from bt.algos.core import Algo, AlgoStack


class DummyAlgo(Algo):
    def __init__(self, return_value=True):
        super().__init__()
        self.return_value = return_value
        self.called = False

    def __call__(self, target):
        self.called = True
        return self.return_value


class HelperAlgo(Algo):
    def __call__(self, target):
        return True


def test_algo_name_defaults_to_class_name():
    algo = HelperAlgo()

    assert algo.name == "HelperAlgo"
    assert algo.run_always is False


def test_market_data_timestamp_prefers_last_day():
    algo = HelperAlgo()
    universe = pd.DataFrame(
        {"A": [100.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    target = SimpleNamespace(
        temp={},
        universe=universe,
        now=universe.index[1],
        last_day=universe.index[0],
    )

    assert algo._resolve_market_data_now(target) == universe.index[0]
    assert algo._resolve_selection_context(target) == ({}, universe, universe.index[0])


def test_market_data_timestamp_falls_back_to_now():
    algo = HelperAlgo()
    target = SimpleNamespace(now=pd.Timestamp("2024-01-02"))

    assert algo._resolve_market_data_now(target) == pd.Timestamp("2024-01-02")


def test_algo_stack_short_circuits_on_first_failure():
    algo1 = DummyAlgo(return_value=True)
    algo2 = DummyAlgo(return_value=False)
    algo3 = DummyAlgo(return_value=True)

    assert not AlgoStack(algo1, algo2, algo3)(SimpleNamespace())
    assert algo1.called is True
    assert algo2.called is True
    assert algo3.called is False


def test_algo_stack_runs_run_always_algos_after_failure():
    algo1 = DummyAlgo(return_value=False)
    algo2 = DummyAlgo(return_value=True)
    algo2.run_always = True
    algo3 = DummyAlgo(return_value=True)

    assert not AlgoStack(algo1, algo2, algo3)(SimpleNamespace())
    assert algo1.called is True
    assert algo2.called is True
    assert algo3.called is False


def test_filter_tickers_by_current_price_respects_missing_and_negative_rules():
    algo = HelperAlgo()
    now = pd.Timestamp("2024-01-01")
    universe = pd.DataFrame(
        {"A": [100.0], "B": [-1.0], "C": [float("nan")]},
        index=[now],
    )

    assert algo._filter_tickers_by_current_price(
        universe,
        now,
        ["A", "B", "C"],
        include_no_data=False,
        include_negative=False,
    ) == ["A"]
    assert algo._filter_tickers_by_current_price(
        universe,
        now,
        ["A", "B", "C"],
        include_no_data=False,
        include_negative=True,
    ) == ["A", "B"]


def test_resolve_candidate_pool_with_fallback_uses_existing_or_fallback_state():
    algo = HelperAlgo()
    temp = {"selected": ["A", "B"]}

    assert algo._resolve_candidate_pool_with_fallback(temp, lambda: False) == ["A", "B"]

    temp = {}

    def fallback():
        temp["selected"] = ["B", "C"]
        return True

    assert algo._resolve_candidate_pool_with_fallback(
        temp,
        fallback,
        allowed_candidates=["A", "B"],
    ) == ["B"]


def test_resolve_candidate_pool_with_fallback_returns_none_on_bad_payload():
    algo = HelperAlgo()

    assert (
        algo._resolve_candidate_pool_with_fallback(
            {"selected": 123},
            lambda: True,
        )
        is None
    )
