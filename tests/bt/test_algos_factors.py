import pandas as pd
import pytest

from bt.core import Strategy
from bt.algos.factors import (
    ExponentialWeightedMovingAverage,
    KernelMovingAverage,
    SetFactor,
    SimpleMovingAverage,
    TotalReturn,
)


def test_simple_moving_average_mean_and_median():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, data={"c1": [1.0, 2.0, 3.0], "c2": [2.0, 4.0, 8.0]})
    s.setup(data)
    s.update(dts[-1])

    algo_mean = SimpleMovingAverage(lookback=pd.DateOffset(days=2), measure="mean")
    assert algo_mean(s)
    assert s.temp["moving_average"]["c1"] == pytest.approx(2.0)
    assert s.temp["moving_average"]["c2"] == pytest.approx(14.0 / 3.0)

    algo_median = SimpleMovingAverage(lookback=pd.DateOffset(days=2), measure="median")
    assert algo_median(s)
    assert s.temp["moving_average"]["c1"] == pytest.approx(2.0)
    assert s.temp["moving_average"]["c2"] == pytest.approx(4.0)


def test_simple_moving_average_uses_selected_when_present():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, data={"c1": [1.0, 2.0], "c2": [2.0, 4.0]})
    s.setup(data)
    s.update(dts[-1])
    s.temp["selected"] = ["c2"]

    algo = SimpleMovingAverage(lookback=pd.DateOffset(days=1))
    assert algo(s)
    assert list(s.temp["moving_average"].index) == ["c2"]


def test_exponential_weighted_moving_average_validates_half_life():
    with pytest.raises(TypeError, match="`half_life`"):
        ExponentialWeightedMovingAverage(half_life="1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`half_life`"):
        ExponentialWeightedMovingAverage(half_life=0)


def test_kernel_moving_average_matches_weighted_kernel():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=4)
    data = pd.DataFrame(index=dts, data={"c1": [1.0, 2.0, 3.0, 4.0], "c2": [2.0] * 4})
    s.setup(data)
    s.update(dts[-1])
    s.temp["selected"] = ["c1", "c2"]

    algo = KernelMovingAverage(lookback=pd.DateOffset(days=2), kernel_factor=1)
    assert algo(s)

    # Linear kernel for [oldest, ..., newest] => weights [1,2,3] / 6.
    assert s.temp["kernel_moving_average"]["c1"] == pytest.approx(
        (2 + 2 * 3 + 3 * 4) / 6
    )
    assert s.temp["kernel_moving_average"]["c2"] == pytest.approx(2.0)


def test_kernel_moving_average_validates_inputs():
    with pytest.raises(TypeError, match="`lookback`"):
        KernelMovingAverage(lookback=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`kernel_factor` must be >= 0"):
        KernelMovingAverage(lookback=pd.DateOffset(days=3), kernel_factor=-1)


def test_set_factor_validates_inputs():
    with pytest.raises(TypeError, match="`factor_str`"):
        SetFactor(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="`factor_df`"):
        SetFactor("x", factor_df=123)  # type: ignore[arg-type]


def test_set_factor_requires_now_in_stat_index():
    s = Strategy("s")
    dts = pd.to_datetime(["2010-01-01", "2010-01-03", "2010-01-05"])
    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), data={"c1": 100.0}
    )
    stat = pd.DataFrame(index=dts, data={"c1": [1.0, 2.0, 3.0]})
    s.setup(data, my_stat=stat)
    s.update(pd.Timestamp("2010-01-04"))  # not in stat index

    algo = SetFactor("my_stat")
    assert not algo(s)


def test_total_return_empty_selected_sets_empty_series():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, data={"c1": [100.0, 101.0]})
    s.setup(data)
    s.update(dts[-1])
    s.temp["selected"] = []

    algo = TotalReturn(lookback=pd.DateOffset(days=1))
    assert algo(s)
    assert list(s.temp["total_return"].index) == ["c1"]


def test_total_return_populates_stats_from_investable_universe():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, data={"c1": [100.0, 110.0], "c2": [0.0, 10.0]})
    s.setup(data)
    s.update(dts[-1])
    s.temp["selected"] = ["c1", "c2"]

    algo = TotalReturn(lookback=pd.DateOffset(days=1))
    assert algo(s)

    stats_row = algo.stats.loc[dts[-1]]
    assert int(stats_row["TOTAL_COVERED"]) == 1
    assert stats_row["MEAN"] == pytest.approx(0.1)
    assert stats_row["MEDIAN"] == pytest.approx(0.1)
    assert stats_row["25TH"] == pytest.approx(0.1)
    assert stats_row["75TH"] == pytest.approx(0.1)
