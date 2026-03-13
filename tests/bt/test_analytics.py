from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bt.analytics import (
    BacktestSummary,
    MultiSeriesPerformanceStats,
    TimeSeriesPerformanceStats,
)
from bt.algos.flow import RunDaily
from bt.algos.portfolio_ops import Rebalance
from bt.algos.weighting import WeightFixed
from bt.core import Backtest, Strategy


class _BacktestStub:
    def __init__(self, name: str, periods: int = 60) -> None:
        idx = pd.date_range("2025-01-01", periods=periods, freq="B")
        t = np.arange(periods, dtype=float)
        prices = pd.Series(
            (1.0 + 0.0005 + 0.0001 * np.sin(t / 7)).cumprod(),
            index=idx,
            name=name,
        )
        self.name = name
        self.strategy = SimpleNamespace(
            prices=prices,
            data=pd.DataFrame(
                {
                    "price": prices,
                    "value": prices * 100.0,
                    "cash": 0.0,
                    "fees": 0.0,
                    "flows": 0.0,
                },
                index=idx,
            ),
            universe=pd.DataFrame(
                {
                    "a": prices,
                    "b": prices * 0.9,
                },
                index=idx,
            ),
            outlays=pd.DataFrame(
                {"a": [0.0, 100.0, 0.0], "b": [0.0, 0.0, -50.0]},
                index=idx[:3],
            ),
        )
        self.weights = pd.DataFrame({"a": 0.6, "b": 0.4}, index=idx)
        self.security_weights = pd.DataFrame({"a": 0.55, "b": 0.45}, index=idx)
        self.positions = pd.DataFrame({"a": 10.0, "b": 20.0}, index=idx)
        self._transactions = pd.DataFrame(
            {"price": [100.0, 101.0], "quantity": [1.0, -1.0]},
            index=pd.MultiIndex.from_tuples(
                [(idx[0], "a"), (idx[1], "a")],
                names=["Date", "Security"],
            ),
        )

    @property
    def get_transactions(self) -> pd.DataFrame:
        return self._transactions


def _prices() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=500, freq="B")
    t = np.arange(len(idx), dtype=float)
    rets = pd.Series(0.0005 + 0.0002 * np.sin(t / 15.0), index=idx, dtype=float)
    return (1.0 + rets).cumprod().rename("strategy")


def _series(name: str, shift: float = 0.0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    t = np.arange(len(idx), dtype=float)
    rets = pd.Series(0.0004 + shift + 0.00015 * np.sin(t / 13.0), index=idx)
    return (1.0 + rets).cumprod().rename(name)


def test_time_series_performance_stats_core_outputs_exist():
    stats = TimeSeriesPerformanceStats(_prices(), rf=0.02, annualization_factor=252)

    assert isinstance(stats.stats, pd.Series)
    assert "total_return" in stats.stats.index
    assert "incep" in stats.stats.index
    assert "sharpe" in stats.stats.index
    assert "daily_skew" in stats.stats.index
    assert "daily_kurt" in stats.stats.index
    assert "max_drawdown_duration" in stats.stats.index
    assert "var_95" in stats.stats.index
    assert "cvar_95" in stats.stats.index
    assert "hit_rate_daily" in stats.stats.index
    assert "mtd" in stats.stats.index
    assert "qtd" in stats.stats.index


def test_time_series_performance_stats_has_expected_tail_risk_metrics():
    stats = TimeSeriesPerformanceStats(_prices())

    assert "max_drawdown" in stats.stats.index
    assert stats.stats["max_drawdown_duration"] >= 0
    assert np.isfinite(stats.stats["var_95"])
    assert np.isfinite(stats.stats["cvar_95"])
    assert 0.0 <= stats.stats["hit_rate_daily"] <= 1.0


def test_incep_is_total_return_when_history_is_less_than_one_year():
    idx = pd.date_range("2025-01-01", periods=120, freq="B")
    prices = pd.Series((1.0 + 0.001) ** np.arange(1, len(idx) + 1), index=idx)
    obj = TimeSeriesPerformanceStats(prices)

    assert obj.stats["incep"] == pytest.approx(obj.stats["total_return"])


def test_incep_is_cagr_when_history_is_at_least_one_year():
    obj = TimeSeriesPerformanceStats(_prices())

    assert obj.stats["incep"] == pytest.approx(obj.stats["cagr"])


def test_time_series_performance_stats_supports_riskfree_price_series():
    prices = _prices()
    rf_prices = (1.0 + pd.Series(0.00005, index=prices.index)).cumprod()
    obj = TimeSeriesPerformanceStats(prices, rf=rf_prices)

    assert np.isfinite(obj.stats["sharpe"])


def test_time_series_performance_stats_clamps_cagr_and_calmar_on_negative_prices():
    idx = pd.date_range("2025-01-01", periods=2, freq="B")
    prices = pd.Series([100.0, -50.0], index=idx)

    obj = TimeSeriesPerformanceStats(prices)

    assert obj.stats["cagr"] == pytest.approx(0.0)
    assert obj.stats["calmar"] == pytest.approx(0.0)


def test_time_series_performance_stats_validates_inputs():
    with pytest.raises(TypeError, match="prices"):
        TimeSeriesPerformanceStats(pd.DataFrame({"x": [1, 2, 3]}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be empty"):
        TimeSeriesPerformanceStats(pd.Series(dtype=float))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        TimeSeriesPerformanceStats(pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2]))
    with pytest.raises(ValueError, match="annualization_factor"):
        TimeSeriesPerformanceStats(_prices(), annualization_factor=0)


def test_multi_series_performance_stats_builds_members_and_stats_table():
    s1 = _series("a")
    s2 = _series("b", shift=0.0001)
    group = MultiSeriesPerformanceStats(s1, s2)

    assert isinstance(group["a"], TimeSeriesPerformanceStats)
    assert isinstance(group[0], TimeSeriesPerformanceStats)
    assert list(group.stats.columns) == ["a", "b"]
    assert "sharpe" in group.stats.index


def test_multi_series_performance_stats_accepts_dataframe_inputs():
    s1 = _series("a")
    s2 = _series("b")
    frame = pd.concat([s1, s2], axis=1)
    group = MultiSeriesPerformanceStats(frame)

    assert list(group.stats.columns) == ["a", "b"]


def test_multi_series_performance_stats_assigns_names_to_unnamed_series():
    s1 = _series("a")
    unnamed = _series("tmp").rename(None)
    group = MultiSeriesPerformanceStats(s1, unnamed)

    assert "a" in group
    assert "series_0" in group


def test_multi_series_performance_stats_validates_inputs():
    with pytest.raises(ValueError, match="at least one"):
        MultiSeriesPerformanceStats()
    with pytest.raises(TypeError, match="Series or DataFrame"):
        MultiSeriesPerformanceStats(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Duplicate series name"):
        MultiSeriesPerformanceStats(_series("a"), _series("a"))
    with pytest.raises(IndexError, match="out of bounds"):
        MultiSeriesPerformanceStats(_series("a"))[3]


def test_backtest_summary_builds_and_exposes_backtest_views():
    b1 = _BacktestStub("s1")
    b2 = _BacktestStub("s2")
    benchmark = pd.DataFrame(
        {"SPX": [100.0, 101.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="B"),
    )
    summary = BacktestSummary(b1, b2, benchmark=benchmark)

    assert list(summary.stats.columns) == ["s1", "s2", "SPX"]
    assert summary.benchmark.equals(benchmark)
    assert list(summary.prices.columns) == ["s1", "s2", "SPX"]
    assert isinstance(summary.get_weights("s1"), pd.DataFrame)
    assert isinstance(summary.get_security_weights(1), pd.DataFrame)
    assert isinstance(summary.get_data("s1"), pd.DataFrame)
    assert isinstance(summary.get_universe("s1"), pd.DataFrame)
    assert isinstance(summary.get_positions("s1"), pd.DataFrame)
    assert isinstance(summary.get_outlays("s1"), pd.DataFrame)
    assert len(summary.get_outlays("s1")) == 2
    assert isinstance(summary.get_transactions("s1"), pd.DataFrame)
    assert isinstance(summary.get_transactions(0), pd.DataFrame)


def test_backtest_summary_plot_prices_builds_figure(monkeypatch):
    called = {"show": 0}

    def fake_show(self):  # noqa: ANN001
        called["show"] += 1

    monkeypatch.setattr("visualization.figure.Figure.show", fake_show)

    b1 = _BacktestStub("s1")
    b2 = _BacktestStub("s2")
    benchmark = pd.DataFrame(
        {"SPX": [100.0, 101.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="B"),
    )
    summary = BacktestSummary(b1, b2, benchmark=benchmark)

    fig = summary.plot_prices()
    built = fig.build().fig

    assert built is not None
    assert built.layout.title.text == "Cumulative Returns"
    assert len(built.data) == 3
    assert {trace.name for trace in built.data} == {"s1", "s2", "SPX"}
    assert called["show"] == 1


def test_backtest_summary_plot_security_weights_builds_figure(monkeypatch):
    called = {"show": 0}

    def fake_show(self):  # noqa: ANN001
        called["show"] += 1

    monkeypatch.setattr("visualization.figure.Figure.show", fake_show)

    summary = BacktestSummary(
        _BacktestStub("Top"), figi_to_ticker={"A": "AAPL", "B": "MSFT"}
    )

    fig = summary.plot_security_weights("Top")
    built = fig.build().fig

    assert built is not None
    assert built.layout.title.text == "Security Weights - Top"
    assert built.layout.yaxis.tickformat == ".0%"
    assert len(built.data) == 2
    assert {trace.name for trace in built.data} == {"AAPL", "MSFT"}
    assert called["show"] == 1


def test_backtest_summary_validates_constructor_inputs():
    with pytest.raises(ValueError, match="at least one"):
        BacktestSummary()

    bad_name = _BacktestStub("ok")
    bad_name.name = ""
    with pytest.raises(TypeError, match="non-empty string"):
        BacktestSummary(bad_name)

    bad_prices = _BacktestStub("ok")
    bad_prices.strategy.prices = pd.DataFrame({"x": [1.0]})
    with pytest.raises(TypeError, match="pandas Series"):
        BacktestSummary(bad_prices)

    dup1 = _BacktestStub("dup")
    dup2 = _BacktestStub("dup")
    with pytest.raises(ValueError, match="unique"):
        BacktestSummary(dup1, dup2)

    with pytest.raises(TypeError, match="pandas DataFrame"):
        BacktestSummary(_BacktestStub("ok"), benchmark="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty"):
        BacktestSummary(_BacktestStub("ok"), benchmark=pd.DataFrame())


def test_backtest_summary_selector_rejects_negative_and_out_of_bounds_indices():
    summary = BacktestSummary(_BacktestStub("s1"))
    with pytest.raises(IndexError, match="out of range"):
        summary.get_weights(-1)
    with pytest.raises(IndexError, match="out of range"):
        summary.get_security_weights(2)


def test_backtest_summary_selector_rejects_unknown_name():
    summary = BacktestSummary(_BacktestStub("s1"))
    with pytest.raises(KeyError, match="not found"):
        summary.get_transactions("missing")


def test_backtest_summary_integration_helpers_with_real_backtest_run():
    names = ["foo", "bar"]
    dates = pd.date_range("2017-01-01", "2017-12-31", freq=pd.tseries.offsets.BDay())
    n = len(dates)
    rdf = pd.DataFrame(np.zeros((n, len(names))), index=dates, columns=names)

    np.random.seed(1)
    rdf[names[0]] = np.random.normal(loc=0.1 / n, scale=0.2 / np.sqrt(n), size=n)
    rdf[names[1]] = np.random.normal(loc=0.04 / n, scale=0.05 / np.sqrt(n), size=n)
    pdf = 100.0 * np.cumprod(1.0 + rdf)

    strategy = Strategy(
        "static",
        [
            RunDaily(run_on_first_date=True),
            WeightFixed(foo=0.6, bar=0.4),
            Rebalance(),
        ],
    )
    backtest = Backtest(strategy, pdf, integer_positions=False, progress_bar=False)
    backtest.run()
    summary = BacktestSummary(backtest)

    assert isinstance(summary.get_security_weights(), pd.DataFrame)
    assert isinstance(summary.get_transactions(), pd.DataFrame)
    assert isinstance(summary.get_weights(), pd.DataFrame)


def test_backtest_summary_handles_bankrupt_backtests():
    prices = pd.DataFrame(
        {"A": [100.0, -200.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    strategy = Strategy("bankrupt", [WeightFixed(A=1.0), Rebalance()])
    backtest = Backtest(strategy, prices, progress_bar=False)
    backtest.run()

    summary = BacktestSummary(backtest)

    assert summary["bankrupt"].stats["cagr"] == pytest.approx(0.0)
    assert summary["bankrupt"].stats["calmar"] == pytest.approx(0.0)
