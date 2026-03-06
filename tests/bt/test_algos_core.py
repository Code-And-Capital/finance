from unittest import mock

import pandas as pd

from bt.algos.core import Algo, AlgoStack


class DummyAlgo(Algo):
    def __init__(self, return_value=True):
        self.return_value = return_value
        self.called = False

    def __call__(self, target):
        self.called = True
        return self.return_value


def test_algo_name():
    class TestAlgo(Algo):
        pass

    actual = TestAlgo()
    assert actual.name == "TestAlgo"
    assert actual.run_always is False


def test_algo_context_helpers():
    class _Algo(Algo):
        def __call__(self, target):
            return True

    algo = _Algo()
    dts = pd.date_range("2026-01-01", periods=2)
    universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    target = mock.MagicMock()
    target.temp = {}
    target.universe = universe
    target.now = dts[0]

    now = algo._resolve_now_in_universe_or_none(target, universe)
    assert now == dts[0]

    context = algo._resolve_selection_context(target)
    assert context is not None
    temp, ctx_universe, ctx_now = context
    assert isinstance(temp, dict)
    assert ctx_universe is universe
    assert ctx_now == dts[0]


def test_algo_stack():
    algo1 = DummyAlgo(return_value=True)
    algo2 = DummyAlgo(return_value=False)
    algo3 = DummyAlgo(return_value=True)

    target = mock.MagicMock()
    stack = AlgoStack(algo1, algo2, algo3)

    actual = stack(target)
    assert not actual
    assert algo1.called
    assert algo2.called
    assert not algo3.called


def test_algo_stack_runs_marked_algos_after_failure():
    algo1 = DummyAlgo(return_value=False)
    algo2 = DummyAlgo(return_value=True)
    algo2.run_always = True
    algo3 = DummyAlgo(return_value=True)

    target = mock.MagicMock()
    stack = AlgoStack(algo1, algo2, algo3)

    actual = stack(target)
    assert not actual
    assert algo1.called
    assert algo2.called
    assert not algo3.called


def test_algo_stack_does_not_run_unmarked_algos_after_failure():
    algo1 = DummyAlgo(return_value=False)
    algo2 = DummyAlgo(return_value=True)
    algo2.run_always = False
    algo3 = DummyAlgo(return_value=True)

    target = mock.MagicMock()
    stack = AlgoStack(algo1, algo2, algo3)

    actual = stack(target)
    assert not actual
    assert algo1.called
    assert not algo2.called
    assert not algo3.called


class _HelperAlgo(Algo):
    def __call__(self, target):
        return True


def test_filter_tickers_by_current_price_include_no_data_returns_input():
    algo = _HelperAlgo()
    dts = pd.date_range("2026-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    tickers = ["c1", "c2"]

    actual = algo._filter_tickers_by_current_price(
        universe=universe,
        now=dts[0],
        tickers=tickers,
        include_no_data=True,
        include_negative=False,
    )
    assert actual == tickers


def test_filter_tickers_by_current_price_excludes_missing_and_non_positive():
    algo = _HelperAlgo()
    dts = pd.date_range("2026-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    universe.loc[dts[0], "c2"] = 0.0
    universe.loc[dts[0], "c3"] = float("nan")

    actual = algo._filter_tickers_by_current_price(
        universe=universe,
        now=dts[0],
        tickers=["c1", "c2", "c3"],
        include_no_data=False,
        include_negative=False,
    )
    assert actual == ["c1"]


def test_filter_tickers_by_current_price_include_negative_keeps_non_null():
    algo = _HelperAlgo()
    dts = pd.date_range("2026-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    universe.loc[dts[0], "c2"] = -5.0
    universe.loc[dts[0], "c3"] = float("nan")

    actual = algo._filter_tickers_by_current_price(
        universe=universe,
        now=dts[0],
        tickers=["c1", "c2", "c3"],
        include_no_data=False,
        include_negative=True,
    )
    assert actual == ["c1", "c2"]


def test_filter_tickers_by_current_price_returns_empty_on_invalid_loc():
    algo = _HelperAlgo()
    dts = pd.date_range("2026-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)

    actual = algo._filter_tickers_by_current_price(
        universe=universe,
        now="not-a-date",  # invalid indexer
        tickers=["c1"],
        include_no_data=False,
        include_negative=False,
    )
    assert actual == []


def test_resolve_candidate_pool_with_fallback_uses_existing_selected():
    algo = _HelperAlgo()
    temp = {"selected": ["c1", "c2"]}

    fallback_called = {"called": False}

    def _fallback():
        fallback_called["called"] = True
        return True

    actual = algo._resolve_candidate_pool_with_fallback(temp, _fallback)
    assert actual == ["c1", "c2"]
    assert not fallback_called["called"]


def test_resolve_candidate_pool_with_fallback_calls_fallback_on_missing_or_empty():
    algo = _HelperAlgo()
    temp = {}

    def _fallback():
        temp["selected"] = ["c3"]
        return True

    actual = algo._resolve_candidate_pool_with_fallback(temp, _fallback)
    assert actual == ["c3"]

    temp["selected"] = []
    actual = algo._resolve_candidate_pool_with_fallback(temp, _fallback)
    assert actual == ["c3"]


def test_resolve_candidate_pool_with_fallback_returns_none_on_failure():
    algo = _HelperAlgo()
    temp = {}

    def _fallback():
        return False

    assert algo._resolve_candidate_pool_with_fallback(temp, _fallback) is None


def test_resolve_candidate_pool_with_fallback_returns_none_on_malformed_selected():
    algo = _HelperAlgo()
    temp = {"selected": 123}

    def _fallback():
        return True

    assert algo._resolve_candidate_pool_with_fallback(temp, _fallback) is None


def test_resolve_candidate_pool_with_fallback_filters_by_allowed_candidates():
    algo = _HelperAlgo()
    temp = {"selected": ["c2", "c3", "c1"]}

    actual = algo._resolve_candidate_pool_with_fallback(
        temp,
        lambda: True,
        allowed_candidates=["c1", "c2"],
    )
    assert actual == ["c1", "c2"]
