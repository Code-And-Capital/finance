from unittest import mock

import pandas as pd

from bt.utils.selection_utils import (
    filter_tickers_by_current_price,
    intersect_candidates_with_pool,
    resolve_candidate_pool_with_fallback,
    resolve_now_in_universe_or_none,
    resolve_selection_context,
    signal_row_to_bool_mask,
)


def test_resolve_now_in_universe_or_none_returns_timestamp_when_valid():
    dts = pd.date_range("2026-01-01", periods=2)
    universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    target = mock.MagicMock()
    target.now = dts[0]

    assert resolve_now_in_universe_or_none(target, universe) == dts[0]


def test_resolve_now_in_universe_or_none_returns_none_when_invalid():
    dts = pd.date_range("2026-01-01", periods=2)
    universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)

    target = mock.MagicMock()
    target.now = None
    assert resolve_now_in_universe_or_none(target, universe) is None

    target.now = "not-a-date"
    assert resolve_now_in_universe_or_none(target, universe) is None

    target.now = pd.Timestamp("1999-01-01")
    assert resolve_now_in_universe_or_none(target, universe) is None


def test_resolve_now_in_universe_or_none_returns_none_when_now_property_raises():
    dts = pd.date_range("2026-01-01", periods=2)
    universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)

    class _Target:
        @property
        def now(self):
            raise TypeError("bad now accessor")

    assert resolve_now_in_universe_or_none(_Target(), universe) is None


def test_resolve_selection_context_returns_tuple_when_valid():
    dts = pd.date_range("2026-01-01", periods=2)
    universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    target = mock.MagicMock()
    target.temp = {}
    target.universe = universe
    target.now = dts[0]

    context = resolve_selection_context(target)
    assert context is not None
    temp, ctx_universe, now = context
    assert isinstance(temp, dict)
    assert ctx_universe is universe
    assert now == dts[0]


def test_resolve_selection_context_returns_none_when_invalid():
    target = mock.MagicMock(spec=[])
    assert resolve_selection_context(target) is None

    target = mock.MagicMock()
    target.temp = []
    target.universe = pd.DataFrame(index=pd.date_range("2026-01-01", periods=1))
    target.now = pd.Timestamp("2026-01-01")
    assert resolve_selection_context(target) is None


def test_resolve_selection_context_returns_none_when_universe_accessor_raises():
    class _Target:
        temp = {}

        @property
        def universe(self):
            raise TypeError("invalid universe slice")

    assert resolve_selection_context(_Target()) is None


def test_filter_tickers_by_current_price_include_no_data_returns_input():
    dts = pd.date_range("2026-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    tickers = ["c1", "c2"]

    actual = filter_tickers_by_current_price(
        universe=universe,
        now=dts[0],
        tickers=tickers,
        include_no_data=True,
        include_negative=False,
    )
    assert actual == tickers


def test_filter_tickers_by_current_price_excludes_missing_and_non_positive():
    dts = pd.date_range("2026-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    universe.loc[dts[0], "c2"] = 0.0
    universe.loc[dts[0], "c3"] = float("nan")

    actual = filter_tickers_by_current_price(
        universe=universe,
        now=dts[0],
        tickers=["c1", "c2", "c3"],
        include_no_data=False,
        include_negative=False,
    )
    assert actual == ["c1"]


def test_filter_tickers_by_current_price_include_negative_keeps_non_null():
    dts = pd.date_range("2026-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    universe.loc[dts[0], "c2"] = -5.0
    universe.loc[dts[0], "c3"] = float("nan")

    actual = filter_tickers_by_current_price(
        universe=universe,
        now=dts[0],
        tickers=["c1", "c2", "c3"],
        include_no_data=False,
        include_negative=True,
    )
    assert actual == ["c1", "c2"]


def test_resolve_candidate_pool_with_fallback_uses_existing_selected():
    temp = {"selected": ["c1", "c2"]}

    fallback_called = {"called": False}

    def _fallback():
        fallback_called["called"] = True
        return True

    actual = resolve_candidate_pool_with_fallback(temp, _fallback)
    assert actual == ["c1", "c2"]
    assert not fallback_called["called"]


def test_resolve_candidate_pool_with_fallback_calls_fallback_on_missing_or_empty():
    temp = {}

    def _fallback():
        temp["selected"] = ["c3"]
        return True

    actual = resolve_candidate_pool_with_fallback(temp, _fallback)
    assert actual == ["c3"]

    temp["selected"] = []
    actual = resolve_candidate_pool_with_fallback(temp, _fallback)
    assert actual == ["c3"]


def test_resolve_candidate_pool_with_fallback_returns_none_on_failure():
    temp = {}

    def _fallback():
        return False

    assert resolve_candidate_pool_with_fallback(temp, _fallback) is None


def test_resolve_candidate_pool_with_fallback_returns_none_on_malformed_selected():
    temp = {"selected": 123}

    def _fallback():
        return True

    assert resolve_candidate_pool_with_fallback(temp, _fallback) is None


def test_resolve_candidate_pool_with_fallback_filters_by_allowed_candidates():
    temp = {"selected": ["c2", "c3", "c1"]}

    actual = resolve_candidate_pool_with_fallback(
        temp,
        lambda: True,
        allowed_candidates=["c1", "c2"],
    )
    assert actual == ["c1", "c2"]


def test_intersect_candidates_with_pool_preserves_candidate_order():
    candidates = ["c3", "c1", "c2"]
    pool = ["c1", "c2"]
    assert intersect_candidates_with_pool(candidates, pool) == ["c1", "c2"]


def test_signal_row_to_bool_mask_handles_numeric_and_missing():
    row = pd.Series([1, 0, pd.NA], index=["c1", "c2", "c3"])
    mask = signal_row_to_bool_mask(row)
    assert list(mask.index[mask]) == ["c1"]
