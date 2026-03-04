from __future__ import division
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from bt.core import Algo, AlgoStack
from bt.core import (
    Strategy,
    SecurityBase,
    Security,
)
from bt.algos.flow import (
    ClosePositionsAfterDates,
)
from bt.algos.portfolio_ops import Rebalance, RebalanceOverTime, HedgeRisks
from bt.algos.selection import SelectThese
from bt.algos.weighting import (
    WeighEqually,
    WeighSpecified,
    ScaleWeights,
    WeighERC,
    WeighTarget,
    WeighInvVol,
    WeighMeanVar,
    WeighRandomly,
    LimitWeights,
    LimitDeltas,
    TargetVol,
)
from bt.algos.stats import SetStat, StatTotalReturn, UpdateRisk


def test_algo_name():
    class TestAlgo(Algo):
        pass

    actual = TestAlgo()

    assert actual.name == "TestAlgo"


class DummyAlgo(Algo):
    def __init__(self, return_value=True):
        self.return_value = return_value
        self.called = False

    def __call__(self, target):
        self.called = True
        return self.return_value


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


def test_rebalance():
    algo = Rebalance()

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 1}
    assert algo(s)
    assert s.value == 1000
    assert s.capital == 0
    c1 = s["c1"]
    assert c1.value == 1000
    assert c1.position == 10
    assert c1.weight == 1.0

    s.temp["weights"] = {"c2": 1}

    assert algo(s)
    assert s.value == 1000
    assert s.capital == 0
    c2 = s["c2"]
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 1000
    assert c2.position == 10
    assert c2.weight == 1.0


def test_rebalance_with_commissions():
    algo = Rebalance()

    s = Strategy("s")
    s.set_commissions(lambda q, p: 1)

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 1}

    assert algo(s)
    assert s.value == 999
    assert s.capital == 99
    c1 = s["c1"]
    assert c1.value == 900
    assert c1.position == 9
    assert c1.weight == 900 / 999.0

    s.temp["weights"] = {"c2": 1}

    assert algo(s)
    assert s.value == 997
    assert s.capital == 97
    c2 = s["c2"]
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 900
    assert c2.position == 9
    assert c2.weight == 900.0 / 997


def test_rebalance_with_cash():
    algo = Rebalance()

    s = Strategy("s")
    s.set_commissions(lambda q, p: 1)

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 1}
    # set cash amount
    s.temp["cash"] = 0.5

    assert algo(s)
    assert s.value == 999
    assert s.capital == 599
    c1 = s["c1"]
    assert c1.value == 400
    assert c1.position == 4
    assert c1.weight == 400.0 / 999

    s.temp["weights"] = {"c2": 1}
    # change cash amount
    s.temp["cash"] = 0.25

    assert algo(s)
    assert s.value == 997
    assert s.capital == 297
    c2 = s["c2"]
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 700
    assert c2.position == 7
    assert c2.weight == 700.0 / 997


def test_rebalance_updatecount():

    algo = Rebalance()

    s = Strategy("s")
    s.use_integer_positions(False)

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4", "c5"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 0.25, "c2": 0.25, "c3": 0.25, "c4": 0.25}

    update = SecurityBase.update
    SecurityBase._update_call_count = 0

    def side_effect(self, *args, **kwargs):
        SecurityBase._update_call_count += 1
        return update(self, *args, **kwargs)

    with mock.patch.object(SecurityBase, "update", side_effect) as mock_update:
        assert algo(s)

    assert s.value == 1000
    assert s.capital == 0

    # Update is called once when each weighted security is created (4)
    # and once for each security after all allocations are made (4)
    assert SecurityBase._update_call_count == 8

    s.update(dts[1])
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}

    update = SecurityBase.update
    SecurityBase._update_call_count = 0

    def side_effect(self, *args, **kwargs):
        SecurityBase._update_call_count += 1
        return update(self, *args, **kwargs)

    with mock.patch.object(SecurityBase, "update", side_effect) as mock_update:
        assert algo(s)

    # Update is called once for each weighted security before allocation (4)
    # and once for each security after all allocations are made (4)
    assert SecurityBase._update_call_count == 8

    s.update(dts[2])
    s.temp["weights"] = {"c1": 0.25, "c2": 0.25, "c3": 0.25, "c4": 0.25}

    update = SecurityBase.update
    SecurityBase._update_call_count = 0

    def side_effect(self, *args, **kwargs):
        SecurityBase._update_call_count += 1
        return update(self, *args, **kwargs)

    with mock.patch.object(SecurityBase, "update", side_effect) as mock_update:
        assert algo(s)

    # Update is called once for each weighted security before allocation (2)
    # and once for each security after all allocations are made (4)
    assert SecurityBase._update_call_count == 6


def test_weight_equally():
    algo = WeighEqually()

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert "c1" in weights
    assert weights["c1"] == 0.5
    assert "c2" in weights
    assert weights["c2"] == 0.5


def test_weight_specified():
    algo = WeighSpecified(c1=0.6, c2=0.4)

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert "c1" in weights
    assert weights["c1"] == 0.6
    assert "c2" in weights
    assert weights["c2"] == 0.4


def test_scale_weights():
    s = Strategy("s")
    algo = ScaleWeights(-0.5)

    s.temp["weights"] = {"c1": 0.5, "c2": -0.4, "c3": 0}
    assert algo(s)
    assert s.temp["weights"] == {"c1": -0.25, "c2": 0.2, "c3": 0}


@mock.patch.object(WeighERC, "calc_erc_weights")
def test_weigh_erc(mock_erc):
    algo = WeighERC(lookback=pd.DateOffset(days=5))

    mock_erc.return_value = pd.Series({"c1": 0.3, "c2": 0.7})

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.update(dts[4])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    assert mock_erc.called
    rets = mock_erc.call_args[0][0]
    assert len(rets) == 4
    assert "c1" in rets
    assert "c2" in rets

    weights = s.temp["weights"]
    assert len(weights) == 2
    assert weights["c1"] == 0.3
    assert weights["c2"] == 0.7


def test_weigh_target():
    algo = WeighTarget("target")

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    target = pd.DataFrame(index=dts[:2], columns=["c1", "c2"], data=0.5)
    target.loc[dts[1], "c1"] = 1.0
    target.loc[dts[1], "c2"] = 0.0

    s.setup(data, target=target)

    s.update(dts[0])
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert weights["c1"] == 0.5
    assert weights["c2"] == 0.5

    s.update(dts[1])
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert weights["c1"] == 1.0
    assert weights["c2"] == 0.0

    s.update(dts[2])
    assert not algo(s)


def test_weigh_inv_vol():
    algo = WeighInvVol(lookback=pd.DateOffset(days=5))

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    # high vol c1
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[2], "c1"] = 95
    data.loc[dts[3], "c1"] = 105
    data.loc[dts[4], "c1"] = 95

    # low vol c2
    data.loc[dts[1], "c2"] = 100.1
    data.loc[dts[2], "c2"] = 99.9
    data.loc[dts[3], "c2"] = 100.1
    data.loc[dts[4], "c2"] = 99.9

    s.setup(data)
    s.update(dts[4])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert weights["c2"] > weights["c1"]
    assert weights["c1"] == pytest.approx(0.020, 3)
    assert weights["c2"] == pytest.approx(0.980, 3)


@mock.patch.object(WeighMeanVar, "calc_mean_var_weights")
def test_weigh_mean_var(mock_mv):
    algo = WeighMeanVar(lookback=pd.DateOffset(days=5))

    mock_mv.return_value = pd.Series({"c1": 0.3, "c2": 0.7})

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.update(dts[4])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    assert mock_mv.called
    rets = mock_mv.call_args[0][0]
    assert len(rets) == 4
    assert "c1" in rets
    assert "c2" in rets

    weights = s.temp["weights"]
    assert len(weights) == 2
    assert weights["c1"] == 0.3
    assert weights["c2"] == 0.7


def test_weigh_randomly():
    s = Strategy("s")
    s.temp["selected"] = ["c1", "c2", "c3"]

    algo = WeighRandomly()
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 3
    assert sum(weights.values()) == 1.0

    algo = WeighRandomly((0.3, 0.5), 0.95)
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 3
    assert sum(weights.values()) == pytest.approx(0.95)
    for c in s.temp["selected"]:
        assert weights[c] <= 0.5
        assert weights[c] >= 0.3


def test_set_stat():
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    stat = pd.DataFrame(index=dts, columns=["c1", "c2"], data=4.0)
    stat.loc[dts[1], "c1"] = 5.0
    stat.loc[dts[1], "c2"] = 6.0

    algo = SetStat("test_stat")

    s.setup(data, test_stat=stat)
    s.update(dts[0])
    print()
    print(s.get_data("test_stat"))
    assert algo(s)
    stat = s.temp["stat"]
    assert stat["c1"] == 4.0
    assert stat["c2"] == 4.0

    s.update(dts[1])
    assert algo(s)
    stat = s.temp["stat"]
    assert stat["c1"] == 5.0
    assert stat["c2"] == 6.0


def test_set_stat_legacy():
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95

    stat = pd.DataFrame(index=dts, columns=["c1", "c2"], data=4.0)
    stat.loc[dts[1], "c1"] = 5.0
    stat.loc[dts[1], "c2"] = 6.0

    algo = SetStat(stat)

    s.setup(data)
    s.update(dts[0])
    assert algo(s)
    stat = s.temp["stat"]
    assert stat["c1"] == 4.0
    assert stat["c2"] == 4.0

    s.update(dts[1])
    assert algo(s)
    stat = s.temp["stat"]
    assert stat["c1"] == 5.0
    assert stat["c2"] == 6.0


def test_stat_total_return():
    algo = StatTotalReturn(lookback=pd.DateOffset(days=3))

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[2], "c1"] = 105
    data.loc[dts[2], "c2"] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    stat = s.temp["stat"]
    assert len(stat) == 2
    assert stat["c1"] == 105.0 / 100 - 1
    assert stat["c2"] == 95.0 / 100 - 1


def test_limit_weights():

    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.temp["weights"] = {"c1": 0.6, "c2": 0.2, "c3": 0.2}

    algo = LimitWeights(0.5)
    assert algo(s)
    w = s.temp["weights"]
    assert w["c1"] == 0.5
    assert w["c2"] == 0.25
    assert w["c3"] == 0.25

    s.temp["weights"] = {"c1": 0.6, "c2": 0.2, "c3": 0.2}
    algo = LimitWeights(0.3)
    assert algo(s)
    w = s.temp["weights"]
    assert w == {}

    s.temp["weights"] = {"c1": 0.4, "c2": 0.3, "c3": 0.3}
    algo = LimitWeights(0.5)
    assert algo(s)
    w = s.temp["weights"]
    assert w["c1"] == 0.4
    assert w["c2"] == 0.3
    assert w["c3"] == 0.3


# def test_limit_deltas():
#     s = Strategy("s")
#     dts = pd.date_range("2010-01-01", periods=3)
#     data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

#     s.setup(data)
#     s.temp["weights"] = {"c1": 1}

#     s.temp["weights"] = {"c1": 0.5, "c2": 0.5}
#     algo = LimitDeltas(0.1)
#     assert algo(s)
#     w = s.temp["weights"]
#     assert len(w) == 2
#     assert w["c1"] == 0.1
#     assert w["c2"] == 0.1

#     s.temp["weights"] = {"c1": 0.5, "c2": -0.5}
#     algo = LimitDeltas(0.1)
#     assert algo(s)
#     w = s.temp["weights"]
#     assert len(w) == 2
#     assert w["c1"] == 0.1
#     assert w["c2"] == -0.1

#     s.temp["weights"] = {"c1": 0.5, "c2": -0.5}
#     algo = LimitDeltas({"c1": 0.1})
#     assert algo(s)
#     w = s.temp["weights"]
#     assert len(w) == 2
#     assert w["c1"] == 0.1
#     assert w["c2"] == -0.5

#     s.temp["weights"] = {"c1": 0.5, "c2": -0.5}
#     algo = LimitDeltas({"c1": 0.1, "c2": 0.3})
#     assert algo(s)
#     w = s.temp["weights"]
#     assert len(w) == 2
#     assert w["c1"] == 0.1
#     assert w["c2"] == -0.3

#     # set exisitng weight
#     s.children["c1"] = SecurityBase("c1")
#     s.children["c1"]._weight = 0.3
#     s.children["c2"] = SecurityBase("c2")
#     s.children["c2"]._weight = -0.7

#     s.temp["weights"] = {"c1": 0.5, "c2": -0.5}
#     algo = LimitDeltas(0.1)
#     assert algo(s)
#     w = s.temp["weights"]
#     assert len(w) == 2
#     assert w["c1"] == 0.4
#     assert w["c2"] == -0.6


def test_rebalance_over_time():
    target = mock.MagicMock()
    rb = mock.MagicMock()

    algo = RebalanceOverTime(n=2)
    # patch in rb function
    algo._rb = rb

    target.temp = {}
    target.temp["weights"] = {"a": 1, "b": 0}

    a = mock.MagicMock()
    a.weight = 0.0
    b = mock.MagicMock()
    b.weight = 1.0
    target.children = {"a": a, "b": b}

    assert algo(target)
    w = target.temp["weights"]
    assert len(w) == 2
    assert w["a"] == 0.5
    assert w["b"] == 0.5

    assert rb.called
    called_tgt = rb.call_args[0][0]
    called_tgt_w = called_tgt.temp["weights"]
    assert len(called_tgt_w) == 2
    assert called_tgt_w["a"] == 0.5
    assert called_tgt_w["b"] == 0.5

    # update weights for next call
    a.weight = 0.5
    b.weight = 0.5

    # clear out temp - same as would Strategy
    target.temp = {}

    assert algo(target)
    w = target.temp["weights"]
    assert len(w) == 2
    assert w["a"] == 1.0
    assert w["b"] == 0.0

    assert rb.call_count == 2

    # update weights for next call
    # should do nothing now
    a.weight = 1
    b.weight = 0

    # clear out temp - same as would Strategy
    target.temp = {}

    assert algo(target)
    # no diff in call_count since last time
    assert rb.call_count == 2


def test_TargetVol():

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=7)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    # high vol c1
    data.loc[dts[0], "c1"] = 95
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[2], "c1"] = 95
    data.loc[dts[3], "c1"] = 105
    data.loc[dts[4], "c1"] = 95
    data.loc[dts[5], "c1"] = 105
    data.loc[dts[6], "c1"] = 95

    # low vol c2
    data.loc[dts[0], "c2"] = 99
    data.loc[dts[1], "c2"] = 101
    data.loc[dts[2], "c2"] = 99
    data.loc[dts[3], "c2"] = 101
    data.loc[dts[4], "c2"] = 99
    data.loc[dts[5], "c2"] = 101
    data.loc[dts[6], "c2"] = 99

    targetVolAlgo = TargetVol(
        0.1,
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=1),
        covar_method="standard",
        annualization_factor=1,
    )

    s.setup(data)
    s.update(dts[6])
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}

    assert targetVolAlgo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert np.isclose(weights["c2"], weights["c1"])

    unannualized_c2_weight = weights["c1"]

    targetVolAlgo = TargetVol(
        0.1 * np.sqrt(252),
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=1),
        covar_method="standard",
        annualization_factor=252,
    )

    s.setup(data)
    s.update(dts[6])
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}

    assert targetVolAlgo(s)
    weights = s.temp["weights"]
    assert len(weights) == 2
    assert np.isclose(weights["c2"], weights["c1"])

    assert np.isclose(unannualized_c2_weight, weights["c2"])


def test_close_positions_after_date():
    c1 = Security("c1")
    c2 = Security("c2")
    c3 = Security("c3")
    s = Strategy("s", children=[c1, c2, c3])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]

    cutoffs = pd.DataFrame({"date": [dts[1], dts[2]]}, index=["c1", "c2"])

    algo = ClosePositionsAfterDates("cutoffs")

    s.setup(data, cutoffs=cutoffs)

    s.update(dts[0])
    s.transact(100, "c1")
    s.transact(100, "c2")
    s.transact(100, "c3")
    algo(s)
    assert c1.position == 100
    assert c2.position == 100
    assert c3.position == 100

    # Don't run anything on dts[1], even though that's when c1 closes
    s.update(dts[2])
    algo(s)
    assert c1.position == 0
    assert c2.position == 0
    assert c3.position == 100
    assert s.perm["closed"] == set(["c1", "c2"])


def test_close_positions_after_date_accepts_series_source():
    c1 = Security("c1")
    s = Strategy("s", children=[c1])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100)
    c1 = s["c1"]

    cutoffs = pd.Series([dts[1]], index=["c1"], name="date")
    algo = ClosePositionsAfterDates(cutoffs)

    s.setup(data)
    s.update(dts[0])
    s.transact(100, "c1")
    s.update(dts[2])
    assert algo(s)
    assert c1.position == 0


def test_close_positions_after_date_skips_root_update_when_nothing_closed():
    close_dates = pd.Series([pd.Timestamp("2099-01-01")], index=["c1"], name="date")
    algo = ClosePositionsAfterDates(close_dates)

    target = mock.MagicMock()
    target.perm = {}
    target.now = pd.Timestamp("2024-01-01")
    target.children = {"c1": SecurityBase("c1")}
    target.root = mock.MagicMock()

    assert algo(target)
    target.root.update.assert_not_called()


def test_update_risk():
    c1 = Security("c1")
    c2 = Security("c2")
    s = Strategy("s", children=[c1, c2])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95
    c1 = s["c1"]
    c2 = s["c2"]

    algo = UpdateRisk("Test", history=False)

    s.setup(data, unit_risk={"Test": data})
    s.adjust(1000)

    s.update(dts[0])
    assert algo(s)
    assert s.risk["Test"] == 0
    assert c1.risk["Test"] == 0
    assert c2.risk["Test"] == 0

    s.transact(1, "c1")
    s.transact(5, "c2")
    assert algo(s)
    assert s.risk["Test"] == 600
    assert c1.risk["Test"] == 100
    assert c2.risk["Test"] == 500

    s.update(dts[1])
    assert algo(s)
    assert s.risk["Test"] == 105 + 5 * 95
    assert c1.risk["Test"] == 105
    assert c2.risk["Test"] == 5 * 95

    assert not hasattr(s, "risks")
    assert not hasattr(c1, "risks")
    assert not hasattr(c2, "risks")


def test_update_risk_history_1():
    c1 = Security("c1")
    c2 = Security("c2")
    s = Strategy("s", children=[c1, c2])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95
    c1 = s["c1"]
    c2 = s["c2"]

    algo = UpdateRisk("Test", history=1)

    s.setup(data, unit_risk={"Test": data})
    s.adjust(1000)

    s.update(dts[0])
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 0

    s.transact(1, "c1")
    s.transact(5, "c2")
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 600

    s.update(dts[1])
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 600
    assert s.risks["Test"].iloc[1] == 105 + 5 * 95

    assert not hasattr(c1, "risks")
    assert not hasattr(c2, "risks")


def test_update_risk_history_2():
    c1 = Security("c1")
    c2 = Security("c2")
    s = Strategy("s", children=[c1, c2])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[1], "c2"] = 95
    c1 = s["c1"]
    c2 = s["c2"]

    algo = UpdateRisk("Test", history=2)

    s.setup(data, unit_risk={"Test": data})
    s.adjust(1000)

    s.update(dts[0])
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 0
    assert c1.risks["Test"].iloc[0] == 0
    assert c2.risks["Test"].iloc[0] == 0

    s.transact(1, "c1")
    s.transact(5, "c2")
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 600
    assert c1.risks["Test"].iloc[0] == 100
    assert c2.risks["Test"].iloc[0] == 500

    s.update(dts[1])
    assert algo(s)
    assert s.risks["Test"].iloc[0] == 600
    assert c1.risks["Test"].iloc[0] == 100
    assert c2.risks["Test"].iloc[0] == 500
    assert s.risks["Test"].iloc[1] == 105 + 5 * 95
    assert c1.risks["Test"].iloc[1] == 105
    assert c2.risks["Test"].iloc[1] == 5 * 95


def test_hedge_risk():
    c1 = Security("c1")
    c2 = Security("c2")
    c3 = Security("c3")
    s = Strategy("s", children=[c1, c2, c3])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]

    risk1 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk2 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk1["c1"] = 1
    risk1["c2"] = 10
    risk2["c1"] = 2
    risk2["c2"] = 5
    risk2["c3"] = 10

    stack = AlgoStack(
        UpdateRisk("Risk1"),
        UpdateRisk("Risk2"),
        SelectThese(["c2", "c3"]),
        HedgeRisks(["Risk1", "Risk2"]),
        UpdateRisk("Risk1"),
        UpdateRisk("Risk2"),
    )

    s.setup(data, unit_risk={"Risk1": risk1, "Risk2": risk2})
    s.adjust(1000)

    s.update(dts[0])
    s.transact(100, "c1")
    stack(s)

    # Check that risk is hedged!
    assert s.risk["Risk1"] == 0
    assert s.risk["Risk2"] == pytest.approx(0, 13)
    # Check that positions are nonzero (trivial solution)
    assert c1.position == 100
    assert c2.position == -10
    assert c3.position == pytest.approx(-(100 * 2 - 10 * 5) / 10.0, 13)


def test_hedge_risk_nan():
    c1 = Security("c1")
    c2 = Security("c2")
    c3 = Security("c3")
    s = Strategy("s", children=[c1, c2, c3])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]

    risk1 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk2 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk1["c1"] = 1
    risk1["c2"] = 10
    risk2["c1"] = float("nan")
    risk2["c2"] = 5
    risk2["c3"] = 10

    stack = AlgoStack(
        UpdateRisk("Risk1"),
        UpdateRisk("Risk2"),
        SelectThese(["c2", "c3"]),
        HedgeRisks(["Risk1", "Risk2"], throw_nan=False),
    )
    stack_throw = AlgoStack(
        UpdateRisk("Risk1"),
        UpdateRisk("Risk2"),
        SelectThese(["c2", "c3"]),
        HedgeRisks(["Risk1", "Risk2"]),
    )

    s.setup(data, unit_risk={"Risk1": risk1, "Risk2": risk2})
    s.adjust(1000)

    s.update(dts[0])
    s.transact(100, "c1")
    assert stack(s)

    did_throw = False
    try:
        stack_throw(s)
    except ValueError:
        did_throw = True
    assert did_throw


def test_hedge_risk_pseudo_under():
    c1 = Security("c1")
    c2 = Security("c2")
    c3 = Security("c3")
    s = Strategy("s", children=[c1, c2, c3])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]
    c3 = s["c3"]

    risk1 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk2 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk1["c1"] = 1
    risk1["c2"] = 10
    risk2["c1"] = 2
    risk2["c3"] = 10

    stack = AlgoStack(
        UpdateRisk("Risk1"),
        UpdateRisk("Risk2"),
        SelectThese(["c2"]),
        HedgeRisks(["Risk1", "Risk2"], pseudo=True),
        UpdateRisk("Risk1"),
        UpdateRisk("Risk2"),
    )

    s.setup(data, unit_risk={"Risk1": risk1, "Risk2": risk2})
    s.adjust(1000)

    s.update(dts[0])
    s.transact(100, "c1")
    stack(s)

    # Check that risk is hedged!
    assert s.risk["Risk1"] == 0
    assert s.risk["Risk2"] != 0
    # Check that positions are nonzero (trivial solution)
    assert c1.position == 100
    assert c2.position == -10
    assert c3.position == 0


def test_hedge_risk_pseudo_over():
    c1 = Security("c1")
    c2 = Security("c2")
    c3 = Security("c3")
    s = Strategy("s", children=[c1, c2, c3])
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100)
    c1 = s["c1"]
    c2 = s["c2"]
    c2 = s["c2"]
    c3 = s["c3"]

    risk1 = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    risk1["c1"] = 1
    risk1["c2"] = 10
    risk1["c3"] = 10  # Same risk as c2

    stack = AlgoStack(
        UpdateRisk("Risk1"),
        SelectThese(["c2", "c3"]),
        HedgeRisks(["Risk1"], pseudo=True),
        UpdateRisk("Risk1"),
    )

    s.setup(data, unit_risk={"Risk1": risk1})
    s.adjust(1000)

    s.update(dts[0])
    s.transact(100, "c1")
    stack(s)

    # Check that risk is hedged!
    assert s.risk["Risk1"] == 0
    # Check that positions are nonzero and risk is evenly split between hedge instruments
    assert c1.position == 100
    assert c2.position == -5
    assert c3.position == -5
