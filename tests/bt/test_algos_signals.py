import pandas as pd
import pytest

from bt.core import Strategy
from bt.algos.signals import (
    DualMACrossoverSignal,
    MomentumSignal,
    PriceCrossOverSignal,
)


def test_price_crossover_signal_selects_assets_above_reference():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(
        index=dts,
        data={"c1": [100.0, 110.0], "c2": [100.0, 90.0], "c3": [100.0, 105.0]},
    )
    s.setup(data)
    s.update(dts[1])
    s.temp["moving_average"] = pd.Series({"c1": 105.0, "c2": 95.0, "c3": 100.0})

    algo = PriceCrossOverSignal()
    assert algo(s)
    assert s.temp["selected"] == ["c1", "c3"]


def test_price_crossover_signal_returns_false_when_reference_missing():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, data={"c1": [100.0]})
    s.setup(data)
    s.update(dts[0])

    algo = PriceCrossOverSignal(ma_name="missing_ma")
    assert not algo(s)


def test_price_crossover_signal_respects_existing_selected_pool():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, data={"c1": [100.0, 110.0], "c2": [100.0, 90.0]})
    s.setup(data)
    s.update(dts[1])
    s.temp["moving_average"] = pd.Series({"c1": 105.0, "c2": 95.0})
    s.temp["selected"] = ["c2"]

    algo = PriceCrossOverSignal()
    assert algo(s)
    assert s.temp["selected"] == []


def test_dual_ma_crossover_signal_selects_short_over_long():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, data={"c1": [100.0], "c2": [100.0], "c3": [100.0]})
    s.setup(data)
    s.update(dts[0])
    s.temp["ma_short"] = pd.Series({"c1": 102.0, "c2": 98.0, "c3": 101.0})
    s.temp["ma_long"] = pd.Series({"c1": 100.0, "c2": 100.0, "c3": 101.0})

    algo = DualMACrossoverSignal()
    assert algo(s)
    assert s.temp["selected"] == ["c1"]


def test_dual_ma_crossover_signal_returns_false_when_series_missing():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, data={"c1": [100.0]})
    s.setup(data)
    s.update(dts[0])
    s.temp["ma_short"] = pd.Series({"c1": 101.0})

    algo = DualMACrossoverSignal()
    assert not algo(s)


def test_momentum_signal_validates_inputs():
    with pytest.raises(TypeError, match="`n`"):
        MomentumSignal(n=1.5)
    with pytest.raises(ValueError, match="`n`"):
        MomentumSignal(n=0)
    with pytest.raises(TypeError, match="`lookback`"):
        MomentumSignal(n=1, lookback=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="`lag`"):
        MomentumSignal(n=1, lag=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="`sort_descending`"):
        MomentumSignal(n=1, sort_descending=1)  # type: ignore[arg-type]


def test_momentum_signal_selects_top_return_name():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=4)
    data = pd.DataFrame(
        index=dts,
        data={
            "c1": [100.0, 101.0, 103.0, 106.0],
            "c2": [100.0, 99.0, 98.0, 97.0],
        },
    )
    s.setup(data)
    s.update(dts[-1])

    algo = MomentumSignal(n=1, lookback=pd.DateOffset(days=3))
    assert algo(s)
    assert s.temp["selected"] == ["c1"]
