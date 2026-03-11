from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bt.analytics import TimeSeriesPerformanceStats


def _prices() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=500, freq="B")
    t = np.arange(len(idx), dtype=float)
    rets = pd.Series(0.0005 + 0.0002 * np.sin(t / 15.0), index=idx, dtype=float)
    return (1.0 + rets).cumprod().rename("strategy")


def test_backtest_performance_stats_core_outputs_exist():
    stats = TimeSeriesPerformanceStats(_prices(), rf=0.02, annualization_factor=252)
    assert isinstance(stats.stats, pd.Series)

    assert "total_return" in stats.stats.index
    assert "incep" in stats.stats.index
    assert "sharpe" in stats.stats.index
    assert "monthly_sharpe" not in stats.stats.index
    assert "yearly_sharpe" not in stats.stats.index
    assert "monthly_mean_ann" not in stats.stats.index
    assert "monthly_vol_ann" not in stats.stats.index
    assert "yearly_mean" not in stats.stats.index
    assert "yearly_vol" not in stats.stats.index
    assert "monthly_skew" not in stats.stats.index
    assert "monthly_kurt" not in stats.stats.index
    assert "yearly_skew" not in stats.stats.index
    assert "yearly_kurt" not in stats.stats.index
    assert "daily_skew" in stats.stats.index
    assert "daily_kurt" in stats.stats.index
    assert "max_drawdown_duration" in stats.stats.index
    assert "var_95" in stats.stats.index
    assert "cvar_95" in stats.stats.index
    assert "hit_rate_daily" in stats.stats.index
    assert "mtd" in stats.stats.index
    assert "qtd" in stats.stats.index
    assert "incep" in stats.stats.index
    assert "5d" not in stats.stats.index


def test_backtest_performance_stats_has_max_drawdown_stat():
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


def test_backtest_performance_stats_supports_riskfree_price_series():
    prices = _prices()
    rf_prices = (1.0 + pd.Series(0.00005, index=prices.index)).cumprod()
    obj = TimeSeriesPerformanceStats(prices, rf=rf_prices)
    assert np.isfinite(obj.stats["sharpe"])


def test_backtest_performance_stats_validates_inputs():
    with pytest.raises(TypeError, match="prices"):
        TimeSeriesPerformanceStats(pd.DataFrame({"x": [1, 2, 3]}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be empty"):
        TimeSeriesPerformanceStats(pd.Series(dtype=float))
    with pytest.raises(TypeError, match="DatetimeIndex"):
        TimeSeriesPerformanceStats(pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2]))
    with pytest.raises(ValueError, match="annualization_factor"):
        TimeSeriesPerformanceStats(_prices(), annualization_factor=0)
