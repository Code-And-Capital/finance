from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bt.algos.expected_returns import (
    BlendedExpectedReturn,
    EWMAExpectedReturns,
    ExcessReturn,
    ExpectedReturns,
    LogReturn,
    MedianReturn,
    RealizedReturn,
    SimpleReturn,
    TrimmedMeanReturn,
    WinsorizedMeanReturn,
)


class _DummyExpectedReturns(ExpectedReturns):
    def calculate_expected_returns(
        self,
        temp,
        universe,
        now,
        selected,
        returns_history,
    ) -> pd.Series | None:
        return returns_history[selected].mean()


class _InvalidExpectedReturns(ExpectedReturns):
    def calculate_expected_returns(
        self,
        temp,
        universe,
        now,
        selected,
        returns_history,
    ) -> pd.Series | None:
        return None


class _SubsetExpectedReturns(ExpectedReturns):
    def calculate_expected_returns(
        self,
        temp,
        universe,
        now,
        selected,
        returns_history,
    ) -> pd.Series | None:
        out = returns_history[selected].mean()
        return out.loc[["A"]]


def _target_base() -> SimpleNamespace:
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    universe = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "B": [50.0, 50.5, 51.0, 51.5, 52.0, 52.5],
        },
        index=dates,
    )
    return SimpleNamespace(temp={}, universe=universe, now=dates[-1])


def _target_outlier() -> SimpleNamespace:
    dates = pd.date_range("2026-01-01", periods=7, freq="D")
    universe = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0, 203.0],
            "B": [50.0, 50.1, 50.2, 50.3, 50.2, 50.25, 50.3],
        },
        index=dates,
    )
    return SimpleNamespace(
        temp={"selected": ["A", "B"]}, universe=universe, now=dates[-1]
    )


def test_expected_returns_writes_outputs_and_cache():
    target = _target_base()
    target.temp["selected"] = ["A", "B"]
    algo = _DummyExpectedReturns(
        lookback=pd.DateOffset(days=4),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    assert "expected_returns" in target.temp
    assert isinstance(target.temp["expected_returns"], pd.Series)
    assert list(target.temp["expected_returns"].index) == ["A", "B"]
    assert target.now in algo.expected_return_estimations.index


def test_expected_returns_missing_selected_writes_empty_and_returns_false():
    target = _target_base()
    algo = _DummyExpectedReturns(lookback=pd.DateOffset(days=3))

    assert algo(target) is False
    assert target.temp["expected_returns"] == {}


def test_expected_returns_empty_selected_writes_empty_and_returns_false():
    target = _target_base()
    target.temp["selected"] = []
    algo = _DummyExpectedReturns(lookback=pd.DateOffset(days=3))

    assert algo(target) is False
    assert target.temp["expected_returns"] == {}


def test_expected_returns_selected_not_in_universe_writes_empty_and_returns_false():
    target = _target_base()
    target.temp["selected"] = ["Z_NOT_IN_UNIVERSE"]
    algo = _DummyExpectedReturns(lookback=pd.DateOffset(days=3))

    assert algo(target) is False
    assert target.temp["expected_returns"] == {}


def test_expected_returns_returns_false_for_invalid_subclass_output():
    target = _target_base()
    target.temp["selected"] = ["A"]
    algo = _InvalidExpectedReturns(lookback=pd.DateOffset(days=3))

    assert algo(target) is False


def test_expected_returns_syncs_selected_to_output_index():
    target = _target_base()
    target.temp["selected"] = ["A", "B"]
    algo = _SubsetExpectedReturns(lookback=pd.DateOffset(days=3))

    assert algo(target) is True
    assert target.temp["selected"] == ["A"]
    assert list(target.temp["expected_returns"].index) == ["A"]


def test_expected_returns_constructor_validates_offsets():
    with pytest.raises(TypeError, match="lookback"):
        _DummyExpectedReturns(lookback=5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="lag"):
        _DummyExpectedReturns(lag=1)  # type: ignore[arg-type]


def test_simple_return_matches_mean_of_historical_returns():
    target = _target_base()
    target.temp["selected"] = ["A", "B"]
    algo = SimpleReturn(
        lookback=pd.DateOffset(days=4),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["expected_returns"]
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["A", "B"]

    start = target.now - pd.DateOffset(days=4)
    end = target.now
    expected = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
        .mean(axis=0)
    )
    pd.testing.assert_series_equal(out, expected)


def test_log_return_matches_mean_of_log_historical_returns():
    target = _target_base()
    target.temp["selected"] = ["A", "B"]
    algo = LogReturn(
        lookback=pd.DateOffset(days=4),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["expected_returns"]
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["A", "B"]

    start = target.now - pd.DateOffset(days=4)
    end = target.now
    expected = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
        .pipe(lambda df: pd.DataFrame(np.log1p(df), index=df.index, columns=df.columns))
        .mean(axis=0)
    )
    pd.testing.assert_series_equal(out, expected)


def test_median_return_matches_median_of_historical_returns():
    target = _target_base()
    target.temp["selected"] = ["A", "B"]
    algo = MedianReturn(
        lookback=pd.DateOffset(days=4),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["expected_returns"]
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["A", "B"]

    start = target.now - pd.DateOffset(days=4)
    end = target.now
    expected = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
        .median(axis=0)
    )
    pd.testing.assert_series_equal(out, expected)


def test_trimmed_mean_return_matches_manual_quantile_trimming():
    target = _target_outlier()
    algo = TrimmedMeanReturn(
        trim_fraction=0.2,
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=0),
    )
    assert algo(target) is True
    out = target.temp["expected_returns"]
    assert isinstance(out, pd.Series)

    start = target.now - pd.DateOffset(days=5)
    end = target.now
    rets = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
    )
    expected = {}
    for col in ["A", "B"]:
        s = rets[col].dropna()
        lo = float(s.quantile(0.2))
        hi = float(s.quantile(0.8))
        trimmed = s[(s >= lo) & (s <= hi)]
        expected[col] = float(trimmed.mean())
    pd.testing.assert_series_equal(out, pd.Series(expected, dtype=float))


def test_winsorized_mean_return_matches_manual_quantile_clipping():
    target = _target_outlier()
    algo = WinsorizedMeanReturn(
        trim_fraction=0.2,
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=0),
    )
    assert algo(target) is True
    out = target.temp["expected_returns"]
    assert isinstance(out, pd.Series)

    start = target.now - pd.DateOffset(days=5)
    end = target.now
    rets = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
    )
    expected = {}
    for col in ["A", "B"]:
        s = rets[col].dropna()
        lo = float(s.quantile(0.2))
        hi = float(s.quantile(0.8))
        expected[col] = float(s.clip(lower=lo, upper=hi).mean())
    pd.testing.assert_series_equal(out, pd.Series(expected, dtype=float))


def test_outlier_expected_returns_validate_trim_fraction():
    with pytest.raises(TypeError, match="trim_fraction"):
        TrimmedMeanReturn(trim_fraction="0.1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="trim_fraction"):
        TrimmedMeanReturn(trim_fraction=0.5)
    with pytest.raises(ValueError, match="trim_fraction"):
        WinsorizedMeanReturn(trim_fraction=-0.1)


def test_ewma_expected_returns_matches_latest_ewma_mean():
    target = _target_base()
    target.temp["selected"] = ["A", "B"]
    algo = EWMAExpectedReturns(
        alpha=0.4,
        lookback=pd.DateOffset(days=4),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["expected_returns"]
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["A", "B"]

    start = target.now - pd.DateOffset(days=4)
    end = target.now
    rets = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
    )
    expected = rets.ewm(alpha=0.4).mean().iloc[-1]
    pd.testing.assert_series_equal(out, expected)


def test_ewma_expected_returns_validates_alpha_and_halflife():
    with pytest.raises(ValueError, match="exactly one"):
        EWMAExpectedReturns(alpha=0.4, halflife=10)
    with pytest.raises(ValueError, match="exactly one"):
        EWMAExpectedReturns()
    with pytest.raises(ValueError, match="alpha"):
        EWMAExpectedReturns(alpha=0.0)
    with pytest.raises(ValueError, match="halflife"):
        EWMAExpectedReturns(halflife=0)


def test_blended_expected_return_matches_equal_weight_blend():
    target = _target_base()
    target.temp["selected"] = ["A", "B"]
    algo = BlendedExpectedReturn(
        halflives=[2, 4],
        use_log_returns=False,
        lookback=pd.DateOffset(days=4),
        lag=pd.DateOffset(days=0),
    )
    assert algo(target) is True
    out = target.temp["expected_returns"]
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["A", "B"]

    start = target.now - pd.DateOffset(days=4)
    end = target.now
    rets = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
    )
    expected = (
        rets.ewm(halflife=2).mean().iloc[-1].reindex(["A", "B"])
        + rets.ewm(halflife=4).mean().iloc[-1].reindex(["A", "B"])
    ) / 2.0
    expected.name = None
    pd.testing.assert_series_equal(out, expected)


def test_blended_expected_return_validates_inputs():
    with pytest.raises(TypeError, match="halflives"):
        BlendedExpectedReturn(halflives="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="halflives"):
        BlendedExpectedReturn(halflives=[2.5])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="half-lives"):
        BlendedExpectedReturn(halflives=[0])
    with pytest.raises(TypeError, match="use_log_returns"):
        BlendedExpectedReturn(use_log_returns=1)  # type: ignore[arg-type]


def _target_with_benchmark_series() -> SimpleNamespace:
    target = _target_base()
    benchmark = pd.Series(
        [100.0, 100.3, 100.1, 100.6, 100.7, 101.0],
        index=target.universe.index,
        name="SPX",
    )

    class _Target(SimpleNamespace):
        def get_data(self, key: str):
            return self._setup_kwargs[key]

    return _Target(
        temp={"selected": ["A", "B"]},
        universe=target.universe,
        now=target.now,
        _setup_kwargs={"index_wide": benchmark},
    )


def _target_with_benchmark_frame() -> SimpleNamespace:
    target = _target_with_benchmark_series()
    benchmark = target._setup_kwargs["index_wide"].to_frame(name="INDEX")
    target._setup_kwargs["index_wide"] = benchmark
    return target


def test_excess_return_matches_mean_log_excess_returns():
    target = _target_with_benchmark_series()
    algo = ExcessReturn(
        index_data_key="index_wide",
        expected_return_estimator=LogReturn(),
        lookback=pd.DateOffset(days=4),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["expected_returns"]
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["A", "B"]

    start = target.now - pd.DateOffset(days=4)
    end = target.now
    prices = target.universe.loc[start:end, ["A", "B"]]
    benchmark_prices = target.get_data("index_wide").loc[start:end]
    excess = (
        prices.pct_change().iloc[1:].sub(benchmark_prices.pct_change().iloc[1:], axis=0)
    )
    expected = pd.DataFrame(
        np.log1p(excess),
        index=excess.index,
        columns=excess.columns,
    ).mean(axis=0)
    pd.testing.assert_series_equal(out, expected)


def test_excess_return_accepts_one_column_benchmark_frame():
    target = _target_with_benchmark_frame()
    algo = ExcessReturn(
        index_data_key="index_wide",
        expected_return_estimator=LogReturn(),
        lookback=pd.DateOffset(days=4),
        lag=pd.DateOffset(days=0),
    )
    assert algo(target) is True
    assert isinstance(target.temp["expected_returns"], pd.Series)


def test_excess_return_rejects_multi_column_benchmark_frame():
    target = _target_with_benchmark_series()
    base = target._setup_kwargs["index_wide"]
    target._setup_kwargs["index_wide"] = pd.DataFrame({"X": base, "Y": base})
    algo = ExcessReturn(
        index_data_key="index_wide",
        expected_return_estimator=LogReturn(),
    )

    assert algo(target) is False


def test_excess_return_validates_index_data_key():
    with pytest.raises(TypeError, match="index_data_key"):
        ExcessReturn(
            index_data_key="",
            expected_return_estimator=LogReturn(),
        )


def test_excess_return_validates_estimator_type():
    with pytest.raises(TypeError, match="expected_return_estimator"):
        ExcessReturn(
            index_data_key="index_wide",
            expected_return_estimator="simple",  # type: ignore[arg-type]
        )


def test_excess_return_delegates_to_wrapped_estimator():
    target = _target_with_benchmark_series()
    algo = ExcessReturn(
        index_data_key="index_wide",
        expected_return_estimator=SimpleReturn(),
        lookback=pd.DateOffset(days=4),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["expected_returns"]
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["A", "B"]

    start = target.now - pd.DateOffset(days=4)
    end = target.now
    prices = target.universe.loc[start:end, ["A", "B"]]
    benchmark_prices = target.get_data("index_wide").loc[start:end]
    excess = (
        prices.pct_change().iloc[1:].sub(benchmark_prices.pct_change().iloc[1:], axis=0)
    )
    expected = excess.mean(axis=0)
    pd.testing.assert_series_equal(out, expected)


def test_realized_return_matches_mean_on_lookback_lookforward_window():
    target = _target_base()
    target.temp["selected"] = ["A", "B"]
    algo = RealizedReturn(
        expected_return_estimator=SimpleReturn(),
        lookback=pd.DateOffset(days=2),
        lookforward=pd.DateOffset(days=1),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["expected_returns"]
    assert isinstance(out, pd.Series)
    assert list(out.index) == ["A", "B"]

    start = target.now - pd.DateOffset(days=2)
    end = target.now + pd.DateOffset(days=1)
    expected = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
        .mean(axis=0)
    )
    pd.testing.assert_series_equal(out, expected)


def test_realized_return_validates_lookforward_type():
    with pytest.raises(TypeError, match="lookforward"):
        RealizedReturn(
            expected_return_estimator=SimpleReturn(),
            lookforward=5,  # type: ignore[arg-type]
        )


def test_realized_return_validates_estimator_type():
    with pytest.raises(TypeError, match="expected_return_estimator"):
        RealizedReturn(expected_return_estimator="simple")  # type: ignore[arg-type]


def test_realized_return_delegates_to_wrapped_estimator():
    target = _target_base()
    target.temp["selected"] = ["A", "B"]
    algo = RealizedReturn(
        expected_return_estimator=LogReturn(),
        lookback=pd.DateOffset(days=2),
        lookforward=pd.DateOffset(days=1),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["expected_returns"]
    assert isinstance(out, pd.Series)

    start = target.now - pd.DateOffset(days=2)
    end = target.now + pd.DateOffset(days=1)
    expected = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
        .pipe(lambda df: pd.DataFrame(np.log1p(df), index=df.index, columns=df.columns))
        .mean(axis=0)
    )
    pd.testing.assert_series_equal(out, expected)
