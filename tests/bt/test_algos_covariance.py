from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import EmpiricalCovariance as SklearnEmpiricalCovariance
from sklearn.covariance import GraphicalLasso as SklearnGraphicalLasso
from sklearn.covariance import MinCovDet, OAS, ledoit_wolf

from bt.algos.covariance import (
    AnnualizeCovariance,
    Covariance,
    EmpiricalCovariance,
    EWMACovariance,
    ExcessCovariance,
    GARCHCovariance,
    GraphicalLassoCovariance,
    LedoitWolfCovariance,
    LedoitWolfNonLinearCovariance,
    LogCovariance,
    MinCovDetCovariance,
    OASCovariance,
    RealizedCovariance,
    RegimeBlendedCovariance,
    RobustHuberCovariance,
    SemiCovariance,
    SimpleCovariance,
)


class _DummyCovariance(Covariance):
    def calculate_covariance(
        self,
        temp,
        universe,
        now,
        selected,
        returns_history,
    ) -> pd.DataFrame | None:
        return returns_history[selected].cov()


class _InvalidCovariance(Covariance):
    def calculate_covariance(
        self,
        temp,
        universe,
        now,
        selected,
        returns_history,
    ) -> pd.DataFrame | None:
        return None


class _SubsetCovariance(Covariance):
    def calculate_covariance(
        self,
        temp,
        universe,
        now,
        selected,
        returns_history,
    ) -> pd.DataFrame | None:
        full = returns_history[selected].cov()
        return full.loc[["A"], ["A"]]


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


def _target_7() -> SimpleNamespace:
    dates = pd.date_range("2026-01-01", periods=7, freq="D")
    universe = pd.DataFrame(
        {
            "A": [100.0, 102.0, 101.0, 103.0, 104.0, 106.0, 105.0],
            "B": [50.0, 50.5, 51.5, 52.0, 51.0, 53.0, 54.0],
        },
        index=dates,
    )
    return SimpleNamespace(
        temp={"selected": ["A", "B"]}, universe=universe, now=dates[-1]
    )


def _target_8() -> SimpleNamespace:
    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    universe = pd.DataFrame(
        {
            "A": [100.0, 102.0, 101.0, 103.0, 104.0, 106.0, 105.0, 107.0],
            "B": [50.0, 50.5, 51.5, 52.0, 51.0, 53.0, 54.0, 55.0],
        },
        index=dates,
    )
    return SimpleNamespace(
        temp={"selected": ["A", "B"]}, universe=universe, now=dates[-1]
    )


def _target_10() -> SimpleNamespace:
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    universe = pd.DataFrame(
        {
            "A": [100.0, 102.0, 101.0, 103.0, 104.0, 106.0, 105.0, 107.0, 108.0, 109.0],
            "B": [50.0, 50.5, 51.5, 52.0, 51.0, 53.0, 54.0, 55.0, 55.5, 56.0],
        },
        index=dates,
    )
    return SimpleNamespace(
        temp={"selected": ["A", "B"]}, universe=universe, now=dates[-1]
    )


def _target_10_with_benchmark() -> SimpleNamespace:
    target = _target_10()
    benchmark = pd.Series(
        [100.0, 100.3, 100.1, 100.6, 100.7, 101.0, 100.9, 101.1, 101.4, 101.6],
        index=target.universe.index,
        name="SPX",
    )

    class _Target(SimpleNamespace):
        def get_data(self, key: str):
            return self._setup_kwargs[key]

    return _Target(
        temp=target.temp,
        universe=target.universe,
        now=target.now,
        _setup_kwargs={"index_wide": benchmark},
    )


def test_covariance_writes_covariance_and_returns_history():
    target = _target_base()
    target.temp["selected"] = ["A", "B"]
    algo = _DummyCovariance(lookback=pd.DateOffset(days=4), lag=pd.DateOffset(days=0))

    assert algo(target) is True
    assert "covariance" in target.temp
    assert "returns_history" in target.temp
    assert isinstance(target.temp["covariance"], pd.DataFrame)
    assert list(target.temp["covariance"].index) == ["A", "B"]
    assert list(target.temp["covariance"].columns) == ["A", "B"]
    assert target.now in algo.cov_estimations


def test_covariance_missing_selected_writes_empty_dict_and_returns_false():
    target = _target_base()
    algo = _DummyCovariance(lookback=pd.DateOffset(days=3))

    assert algo(target) is False
    assert target.temp["covariance"] == {}
    assert target.temp["returns_history"].empty


def test_covariance_empty_selected_writes_empty_dict_and_returns_false():
    target = _target_base()
    target.temp["selected"] = []
    algo = _DummyCovariance(lookback=pd.DateOffset(days=3))

    assert algo(target) is False
    assert target.temp["covariance"] == {}
    assert target.temp["returns_history"].empty


def test_covariance_selected_not_in_universe_writes_empty_dict_and_returns_false():
    target = _target_base()
    target.temp["selected"] = ["Z_NOT_IN_UNIVERSE"]
    algo = _DummyCovariance(lookback=pd.DateOffset(days=3))

    assert algo(target) is False
    assert target.temp["covariance"] == {}
    assert target.temp["returns_history"].empty


def test_covariance_returns_false_for_invalid_subclass_output():
    target = _target_base()
    target.temp["selected"] = ["A"]
    algo = _InvalidCovariance(lookback=pd.DateOffset(days=3))

    assert algo(target) is False


def test_covariance_syncs_selected_to_covariance_columns():
    target = _target_base()
    target.temp["selected"] = ["A", "B"]
    algo = _SubsetCovariance(lookback=pd.DateOffset(days=3))

    assert algo(target) is True
    assert target.temp["selected"] == ["A"]
    assert list(target.temp["covariance"].columns) == ["A"]


def test_covariance_constructor_validates_offsets():
    with pytest.raises(TypeError, match="lookback"):
        _DummyCovariance(lookback=5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="lag"):
        _DummyCovariance(lag=1)  # type: ignore[arg-type]


def test_annualize_covariance_scales_covariance_matrix():
    target = SimpleNamespace(temp={})
    target.temp["covariance"] = pd.DataFrame(
        [[0.1, 0.02], [0.02, 0.2]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    algo = AnnualizeCovariance(annualization_factor=252)

    assert algo(target) is True
    expected = pd.DataFrame(
        [[25.2, 5.04], [5.04, 50.4]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    pd.testing.assert_frame_equal(target.temp["covariance"], expected)


def test_annualize_covariance_returns_false_when_covariance_missing():
    target = SimpleNamespace(temp={})
    algo = AnnualizeCovariance()
    assert algo(target) is False


def test_annualize_covariance_returns_false_when_covariance_not_dataframe():
    target = SimpleNamespace(temp={"covariance": {}})
    algo = AnnualizeCovariance()
    assert algo(target) is False


def test_annualize_covariance_validates_annualization_factor():
    with pytest.raises(ValueError, match="annualization_factor"):
        AnnualizeCovariance(-1.0)
    with pytest.raises(TypeError, match="annualization_factor"):
        AnnualizeCovariance("252")  # type: ignore[arg-type]


def test_simple_covariance_computes_sample_covariance_ddof_1():
    target = _target_7()
    algo = SimpleCovariance(
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=0),
        ddof=1,
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]

    start = target.now - pd.DateOffset(days=5)
    end = target.now
    expected = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
        .cov(ddof=1)
    )
    pd.testing.assert_frame_equal(out, expected)


def test_simple_covariance_validates_ddof():
    with pytest.raises(TypeError, match="ddof"):
        SimpleCovariance(ddof=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ddof"):
        SimpleCovariance(ddof=-1)


def test_log_covariance_computes_sample_covariance_on_log_returns():
    target = _target_7()
    algo = LogCovariance(
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=0),
        ddof=1,
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]

    start = target.now - pd.DateOffset(days=5)
    end = target.now
    expected = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
        .pipe(lambda df: pd.DataFrame(np.log1p(df), index=df.index, columns=df.columns))
        .cov(ddof=1)
    )
    pd.testing.assert_frame_equal(out, expected)


def test_log_covariance_validates_ddof():
    with pytest.raises(TypeError, match="ddof"):
        LogCovariance(ddof=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ddof"):
        LogCovariance(ddof=-1)


def test_semi_covariance_matches_downside_formula_no_missing():
    target = _target_7()
    algo = SemiCovariance(
        threshold_return=0.0,
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]

    start = target.now - pd.DateOffset(days=5)
    end = target.now
    rets = target.universe.loc[start:end, ["A", "B"]].pct_change().iloc[1:]
    downside = (rets - 0.0).clip(upper=0.0)
    expected = (downside.T @ downside) / len(downside.index)
    pd.testing.assert_frame_equal(out, expected.astype(float))


def test_semi_covariance_handles_pairwise_missing():
    target = _target_7()
    target.universe.loc[target.universe.index[-2], "A"] = pd.NA
    algo = SemiCovariance(
        threshold_return=0.0,
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (2, 2)
    assert out.notna().any().any()


def test_semi_covariance_validates_threshold_return():
    with pytest.raises(TypeError, match="threshold_return"):
        SemiCovariance(threshold_return="0.0")  # type: ignore[arg-type]


def test_ewma_covariance_computes_latest_ewm_covariance_with_alpha():
    target = _target_8()
    algo = EWMACovariance(
        alpha=0.2,
        lookback=pd.DateOffset(days=6),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]

    start = target.now - pd.DateOffset(days=6)
    end = target.now
    returns = target.universe.loc[start:end, ["A", "B"]].pct_change().iloc[1:]
    returns = returns.replace([float("inf"), -float("inf")], pd.NA).astype(float)
    expected_full = returns.ewm(alpha=0.2).cov()
    expected = expected_full.xs(expected_full.index.get_level_values(0)[-1], level=0)
    expected = expected.reindex(index=["A", "B"], columns=["A", "B"])
    pd.testing.assert_frame_equal(out, expected)


def test_ewma_covariance_supports_halflife_parameter():
    target = _target_8()
    algo = EWMACovariance(
        halflife=10,
        lookback=pd.DateOffset(days=6),
        lag=pd.DateOffset(days=0),
    )
    assert algo(target) is True
    assert 0.0 < algo.alpha <= 1.0


def test_ewma_covariance_parameter_validation():
    with pytest.raises(ValueError, match="exactly one"):
        EWMACovariance(alpha=None, halflife=None)
    with pytest.raises(ValueError, match="exactly one"):
        EWMACovariance(alpha=0.5, halflife=10)
    with pytest.raises(ValueError, match="alpha"):
        EWMACovariance(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        EWMACovariance(alpha=1.5)
    with pytest.raises(ValueError, match="halflife"):
        EWMACovariance(halflife=0)


def test_regime_blended_covariance_blends_multiple_ewma_pairs():
    target = _target_10()

    algo = RegimeBlendedCovariance(
        halflife_pairs=[(10, 21), (21, 63)],
        min_coverage=0.0,
        use_log_returns=True,
        lookback=pd.DateOffset(days=8),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    np.testing.assert_allclose(out.to_numpy(), out.to_numpy().T, rtol=1e-10, atol=1e-10)
    assert target.temp["selected"] == ["A", "B"]


def test_regime_blended_covariance_validates_inputs():
    with pytest.raises(TypeError, match="halflife_pairs"):
        RegimeBlendedCovariance(halflife_pairs="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="halflife_pairs"):
        RegimeBlendedCovariance(halflife_pairs=[(10,)])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="half-lives"):
        RegimeBlendedCovariance(halflife_pairs=[(10, 0)])
    with pytest.raises(TypeError, match="min_coverage"):
        RegimeBlendedCovariance(min_coverage="0.8")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_coverage"):
        RegimeBlendedCovariance(min_coverage=-0.1)
    with pytest.raises(ValueError, match="min_coverage"):
        RegimeBlendedCovariance(min_coverage=1.1)
    with pytest.raises(TypeError, match="use_log_returns"):
        RegimeBlendedCovariance(use_log_returns=1)  # type: ignore[arg-type]


def test_oas_covariance_matches_sklearn_oas():
    target = _target_8()
    algo = OASCovariance(
        min_coverage=0.5,
        lookback=pd.DateOffset(days=6),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]

    start = target.now - pd.DateOffset(days=6)
    end = target.now
    returns = target.universe.loc[start:end, ["A", "B"]].pct_change().iloc[1:]
    returns = returns.replace([float("inf"), -float("inf")], pd.NA).astype(float)
    fit_data = returns.dropna(how="any")
    expected = pd.DataFrame(
        OAS().fit(fit_data).covariance_,
        index=["A", "B"],
        columns=["A", "B"],
    )
    pd.testing.assert_frame_equal(out, expected)


def test_oas_covariance_drops_missing_rows_before_fit():
    target = _target_8()
    target.universe.loc[target.universe.index[-2], "A"] = pd.NA
    algo = OASCovariance(
        min_coverage=0.0,
        lookback=pd.DateOffset(days=6),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (2, 2)
    assert target.temp["selected"] == ["A", "B"]


def test_oas_covariance_filters_low_coverage_assets_before_fit():
    target = _target_8()
    target.universe.loc[target.universe.index[-6:], "B"] = pd.NA
    algo = OASCovariance(
        min_coverage=0.6,
        lookback=pd.DateOffset(days=6),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A"]
    assert list(out.columns) == ["A"]
    assert target.temp["selected"] == ["A"]


def test_oas_covariance_validates_min_coverage():
    with pytest.raises(TypeError, match="min_coverage"):
        OASCovariance(min_coverage="0.8")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_coverage"):
        OASCovariance(min_coverage=-0.1)
    with pytest.raises(ValueError, match="min_coverage"):
        OASCovariance(min_coverage=1.1)


def test_ledoit_wolf_covariance_matches_sklearn():
    target = _target_8()
    algo = LedoitWolfCovariance(
        min_coverage=0.5,
        lookback=pd.DateOffset(days=6),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]

    start = target.now - pd.DateOffset(days=6)
    end = target.now
    returns = target.universe.loc[start:end, ["A", "B"]].pct_change().iloc[1:]
    returns = returns.replace([float("inf"), -float("inf")], pd.NA).astype(float)
    fit_data = returns.dropna(how="any")
    expected = pd.DataFrame(
        ledoit_wolf(fit_data)[0],
        index=["A", "B"],
        columns=["A", "B"],
    )
    pd.testing.assert_frame_equal(out, expected)


def test_ledoit_wolf_covariance_filters_low_coverage_assets_before_fit():
    target = _target_8()
    target.universe.loc[target.universe.index[-6:], "B"] = pd.NA
    algo = LedoitWolfCovariance(
        min_coverage=0.6,
        lookback=pd.DateOffset(days=6),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A"]
    assert list(out.columns) == ["A"]
    assert target.temp["selected"] == ["A"]


def test_ledoit_wolf_covariance_validates_min_coverage():
    with pytest.raises(TypeError, match="min_coverage"):
        LedoitWolfCovariance(min_coverage="0.8")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_coverage"):
        LedoitWolfCovariance(min_coverage=-0.1)
    with pytest.raises(ValueError, match="min_coverage"):
        LedoitWolfCovariance(min_coverage=1.1)


def test_ledoit_wolf_nonlinear_covariance_shape_symmetry_and_finite():
    target = _target_8()
    algo = LedoitWolfNonLinearCovariance(
        min_coverage=0.5,
        lookback=pd.DateOffset(days=6),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    np.testing.assert_allclose(out.to_numpy(), out.to_numpy().T, rtol=1e-10, atol=1e-10)


def test_ledoit_wolf_nonlinear_covariance_filters_low_coverage_assets():
    target = _target_8()
    target.universe.loc[target.universe.index[-6:], "B"] = pd.NA
    algo = LedoitWolfNonLinearCovariance(
        min_coverage=0.6,
        lookback=pd.DateOffset(days=6),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A"]
    assert list(out.columns) == ["A"]
    assert target.temp["selected"] == ["A"]


def test_ledoit_wolf_nonlinear_covariance_validates_min_coverage():
    with pytest.raises(TypeError, match="min_coverage"):
        LedoitWolfNonLinearCovariance(min_coverage="0.8")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_coverage"):
        LedoitWolfNonLinearCovariance(min_coverage=-0.1)
    with pytest.raises(ValueError, match="min_coverage"):
        LedoitWolfNonLinearCovariance(min_coverage=1.1)


def test_mincovdet_covariance_matches_sklearn():
    target = _target_10()
    algo = MinCovDetCovariance(
        min_coverage=0.0,
        lookback=pd.DateOffset(days=8),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]

    start = target.now - pd.DateOffset(days=8)
    end = target.now
    returns = target.universe.loc[start:end, ["A", "B"]].pct_change().iloc[1:]
    returns = returns.replace([float("inf"), -float("inf")], pd.NA).astype(float)
    fit_data = returns.dropna(how="any")
    expected = pd.DataFrame(
        MinCovDet().fit(fit_data).covariance_,
        index=["A", "B"],
        columns=["A", "B"],
    )
    pd.testing.assert_frame_equal(out, expected)


def test_mincovdet_covariance_filters_low_coverage_assets():
    target = _target_10()
    target.universe.loc[target.universe.index[-7:], "B"] = pd.NA
    algo = MinCovDetCovariance(
        min_coverage=0.6,
        lookback=pd.DateOffset(days=8),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A"]
    assert list(out.columns) == ["A"]
    assert target.temp["selected"] == ["A"]


def test_mincovdet_covariance_validates_min_coverage():
    with pytest.raises(TypeError, match="min_coverage"):
        MinCovDetCovariance(min_coverage="0.8")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_coverage"):
        MinCovDetCovariance(min_coverage=-0.1)
    with pytest.raises(ValueError, match="min_coverage"):
        MinCovDetCovariance(min_coverage=1.1)


def test_empirical_covariance_matches_sklearn_on_log_returns():
    target = _target_10()
    algo = EmpiricalCovariance(
        min_coverage=0.0,
        use_log_returns=True,
        lookback=pd.DateOffset(days=8),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]

    start = target.now - pd.DateOffset(days=8)
    end = target.now
    returns = target.universe.loc[start:end, ["A", "B"]].pct_change().iloc[1:]
    returns = returns.replace([float("inf"), -float("inf")], pd.NA).astype(float)
    fit_data = np.log1p(returns.dropna(how="any"))
    expected = pd.DataFrame(
        SklearnEmpiricalCovariance().fit(fit_data).covariance_,
        index=["A", "B"],
        columns=["A", "B"],
    )
    pd.testing.assert_frame_equal(out, expected)


def test_empirical_covariance_filters_low_coverage_assets():
    target = _target_10()
    target.universe.loc[target.universe.index[-7:], "B"] = pd.NA
    algo = EmpiricalCovariance(
        min_coverage=0.6,
        use_log_returns=True,
        lookback=pd.DateOffset(days=8),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A"]
    assert list(out.columns) == ["A"]
    assert target.temp["selected"] == ["A"]


def test_empirical_covariance_validates_inputs():
    with pytest.raises(TypeError, match="min_coverage"):
        EmpiricalCovariance(min_coverage="0.8")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_coverage"):
        EmpiricalCovariance(min_coverage=-0.1)
    with pytest.raises(ValueError, match="min_coverage"):
        EmpiricalCovariance(min_coverage=1.1)
    with pytest.raises(TypeError, match="use_log_returns"):
        EmpiricalCovariance(use_log_returns=1)  # type: ignore[arg-type]


def test_realized_covariance_uses_lookforward_window():
    target = _target_10()
    algo = RealizedCovariance(
        covariance_estimator=SimpleCovariance(ddof=1),
        lookback=pd.DateOffset(days=3),
        lookforward=pd.DateOffset(days=2),
        lag=pd.DateOffset(days=1),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]

    now = target.now
    start = now - pd.DateOffset(days=3)
    end = (now - pd.DateOffset(days=1)) + pd.DateOffset(days=2)
    expected = (
        target.universe.loc[start:end, ["A", "B"]]
        .pct_change()
        .iloc[1:]
        .replace([float("inf"), -float("inf")], pd.NA)
        .astype(float)
        .cov(ddof=1)
    )
    pd.testing.assert_frame_equal(out, expected)


def test_realized_covariance_validates_inputs():
    with pytest.raises(TypeError, match="covariance_estimator"):
        RealizedCovariance(covariance_estimator="simple")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="lookforward"):
        RealizedCovariance(
            covariance_estimator=SimpleCovariance(),
            lookforward=1,  # type: ignore[arg-type]
        )


def test_excess_covariance_matches_excess_log_covariance():
    target = _target_10_with_benchmark()
    algo = ExcessCovariance(
        index_data_key="index_wide",
        covariance_estimator=LogCovariance(ddof=1),
        lookback=pd.DateOffset(days=8),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]

    start = target.now - pd.DateOffset(days=8)
    end = target.now
    prices = target.universe.loc[start:end, ["A", "B"]]
    benchmark = target.get_data("index_wide").loc[start:end]
    excess = prices.pct_change().iloc[1:].sub(benchmark.pct_change().iloc[1:], axis=0)
    expected = np.log1p(excess).replace([np.inf, -np.inf], np.nan).cov(ddof=1)
    pd.testing.assert_frame_equal(out, expected)


def test_excess_covariance_returns_false_when_benchmark_missing():
    target = _target_10()
    target.get_data = lambda _key: (_ for _ in ()).throw(KeyError("missing"))
    algo = ExcessCovariance(
        index_data_key="index_wide",
        covariance_estimator=SimpleCovariance(),
    )
    assert algo(target) is False


def test_excess_covariance_validates_inputs():
    with pytest.raises(TypeError, match="index_data_key"):
        ExcessCovariance(
            index_data_key=1,  # type: ignore[arg-type]
            covariance_estimator=SimpleCovariance(),
        )
    with pytest.raises(TypeError, match="covariance_estimator"):
        ExcessCovariance(
            index_data_key="index_wide",
            covariance_estimator="simple",  # type: ignore[arg-type]
        )


def test_garch_covariance_with_mocked_mgarch(monkeypatch):
    target = _target_10()

    class _FakeModel:
        def __init__(self, dist: str) -> None:
            self.dist = dist

        def fit(self, data):
            self._fit_data = data
            return self

        def predict(self, ndays: int):
            _ = ndays
            return {"cov": np.array([[0.10, 0.01], [0.01, 0.20]])}

    import bt.algos.covariance.garch as garch_module

    monkeypatch.setattr(
        garch_module.mgarch,
        "mgarch",
        lambda dist="norm": _FakeModel(dist=dist),
    )

    algo = GARCHCovariance(
        distribution="norm",
        forecast_period=21,
        min_coverage=0.0,
        use_log_returns=True,
        lookback=pd.DateOffset(days=8),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]
    pd.testing.assert_frame_equal(
        out,
        pd.DataFrame(
            [[0.10, 0.01], [0.01, 0.20]], index=["A", "B"], columns=["A", "B"]
        ),
    )


def test_garch_covariance_returns_false_when_mgarch_fit_raises(monkeypatch):
    target = _target_10()

    class _BrokenModel:
        def fit(self, _data):
            raise RuntimeError("fit failed")

    import bt.algos.covariance.garch as garch_module

    monkeypatch.setattr(
        garch_module.mgarch,
        "mgarch",
        lambda dist="norm": _BrokenModel(),
    )

    algo = GARCHCovariance(min_coverage=0.0, lookback=pd.DateOffset(days=8))
    assert algo(target) is False


def test_garch_covariance_validates_inputs():
    with pytest.raises(ValueError, match="distribution"):
        GARCHCovariance(distribution="gaussian")
    with pytest.raises(ValueError, match="forecast_period"):
        GARCHCovariance(forecast_period=0)
    with pytest.raises(TypeError, match="forecast_period"):
        GARCHCovariance(forecast_period=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="min_coverage"):
        GARCHCovariance(min_coverage="0.8")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_coverage"):
        GARCHCovariance(min_coverage=-0.1)
    with pytest.raises(ValueError, match="min_coverage"):
        GARCHCovariance(min_coverage=1.1)
    with pytest.raises(TypeError, match="use_log_returns"):
        GARCHCovariance(use_log_returns=1)  # type: ignore[arg-type]


def test_graphical_lasso_covariance_matches_sklearn():
    target = _target_10()
    algo = GraphicalLassoCovariance(
        alpha=0.02,
        min_coverage=0.0,
        use_log_returns=True,
        max_iter=300,
        tol=1e-4,
        lookback=pd.DateOffset(days=8),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]

    start = target.now - pd.DateOffset(days=8)
    end = target.now
    returns = target.universe.loc[start:end, ["A", "B"]].pct_change().iloc[1:]
    returns = returns.replace([float("inf"), -float("inf")], pd.NA).astype(float)
    fit_data = np.log1p(returns.dropna(how="any"))
    expected = SklearnGraphicalLasso(alpha=0.02, max_iter=300, tol=1e-4)
    expected.fit(fit_data.to_numpy(dtype=float))
    np.testing.assert_allclose(
        out.to_numpy(), expected.covariance_, rtol=1e-6, atol=1e-8
    )


def test_graphical_lasso_covariance_filters_low_coverage_assets():
    target = _target_10()
    target.universe.loc[target.universe.index[-7:], "B"] = pd.NA
    algo = GraphicalLassoCovariance(
        alpha=0.02,
        min_coverage=0.6,
        lookback=pd.DateOffset(days=8),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A"]
    assert list(out.columns) == ["A"]
    assert target.temp["selected"] == ["A"]


def test_graphical_lasso_covariance_validates_inputs():
    with pytest.raises(TypeError, match="alpha"):
        GraphicalLassoCovariance(alpha="0.01")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha"):
        GraphicalLassoCovariance(alpha=0.0)
    with pytest.raises(TypeError, match="min_coverage"):
        GraphicalLassoCovariance(min_coverage="0.8")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_coverage"):
        GraphicalLassoCovariance(min_coverage=-0.1)
    with pytest.raises(ValueError, match="min_coverage"):
        GraphicalLassoCovariance(min_coverage=1.1)
    with pytest.raises(TypeError, match="use_log_returns"):
        GraphicalLassoCovariance(use_log_returns=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="max_iter"):
        GraphicalLassoCovariance(max_iter=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_iter"):
        GraphicalLassoCovariance(max_iter=0)
    with pytest.raises(TypeError, match="tol"):
        GraphicalLassoCovariance(tol="1e-4")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tol"):
        GraphicalLassoCovariance(tol=0.0)


def test_robust_huber_covariance_shape_symmetry_and_finite():
    target = _target_10()
    # Inject a large outlier in one row to stress robustness.
    target.universe.loc[target.universe.index[-2], "A"] = 200.0
    algo = RobustHuberCovariance(
        c=1.345,
        min_coverage=0.0,
        lookback=pd.DateOffset(days=8),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A", "B"]
    assert list(out.columns) == ["A", "B"]
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    np.testing.assert_allclose(out.to_numpy(), out.to_numpy().T, rtol=1e-10, atol=1e-10)


def test_robust_huber_covariance_filters_low_coverage_assets():
    target = _target_10()
    target.universe.loc[target.universe.index[-7:], "B"] = pd.NA
    algo = RobustHuberCovariance(
        min_coverage=0.6,
        lookback=pd.DateOffset(days=8),
        lag=pd.DateOffset(days=0),
    )

    assert algo(target) is True
    out = target.temp["covariance"]
    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == ["A"]
    assert list(out.columns) == ["A"]
    assert target.temp["selected"] == ["A"]


def test_robust_huber_covariance_validates_inputs():
    with pytest.raises(TypeError, match="c"):
        RobustHuberCovariance(c="1.345")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="c"):
        RobustHuberCovariance(c=0.0)
    with pytest.raises(TypeError, match="min_coverage"):
        RobustHuberCovariance(min_coverage="0.8")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_coverage"):
        RobustHuberCovariance(min_coverage=-0.1)
    with pytest.raises(ValueError, match="min_coverage"):
        RobustHuberCovariance(min_coverage=1.1)
    with pytest.raises(TypeError, match="max_iter"):
        RobustHuberCovariance(max_iter=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_iter"):
        RobustHuberCovariance(max_iter=0)
    with pytest.raises(TypeError, match="tol"):
        RobustHuberCovariance(tol="1e-6")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tol"):
        RobustHuberCovariance(tol=0.0)
