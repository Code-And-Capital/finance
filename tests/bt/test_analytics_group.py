from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bt.analytics import MultiSeriesPerformanceStats, TimeSeriesPerformanceStats


def _series(name: str, shift: float = 0.0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    t = np.arange(len(idx), dtype=float)
    rets = pd.Series(0.0004 + shift + 0.00015 * np.sin(t / 13.0), index=idx)
    return (1.0 + rets).cumprod().rename(name)


def test_group_performance_stats_builds_members_and_stats_table():
    s1 = _series("a")
    s2 = _series("b", shift=0.0001)
    group = MultiSeriesPerformanceStats(s1, s2)

    assert isinstance(group["a"], TimeSeriesPerformanceStats)
    assert isinstance(group[0], TimeSeriesPerformanceStats)
    assert list(group.stats.columns) == ["a", "b"]
    assert "sharpe" in group.stats.index


def test_group_performance_stats_accepts_dataframe_inputs():
    s1 = _series("a")
    s2 = _series("b")
    frame = pd.concat([s1, s2], axis=1)
    group = MultiSeriesPerformanceStats(frame)
    assert list(group.stats.columns) == ["a", "b"]


def test_group_performance_stats_assigns_names_to_unnamed_series():
    s1 = _series("a")
    unnamed = _series("tmp").rename(None)
    group = MultiSeriesPerformanceStats(s1, unnamed)
    assert "a" in group
    assert "series_0" in group


def test_group_performance_stats_validates_inputs():
    with pytest.raises(ValueError, match="at least one"):
        MultiSeriesPerformanceStats()
    with pytest.raises(TypeError, match="Series or DataFrame"):
        MultiSeriesPerformanceStats(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Duplicate series name"):
        MultiSeriesPerformanceStats(_series("a"), _series("a"))
    with pytest.raises(IndexError, match="out of bounds"):
        MultiSeriesPerformanceStats(_series("a"))[3]
