from __future__ import division
from datetime import datetime, timedelta
from unittest import mock

import sys
import os

import numpy as np
import pandas as pd
import pytest
import random

from bt.core import Algo, AlgoStack
from bt.core import (
    Strategy,
    SecurityBase,
    Security,
    CouponPayingSecurity,
    HedgeSecurity,
    CouponPayingHedgeSecurity,
    FixedIncomeSecurity,
)
from bt.algos.debugging import PrintTempData, PrintInfo
from bt.algos.flow import (
    RunOnce,
    RunPeriod,
    RunDaily,
    RunWeekly,
    RunMonthly,
    RunQuarterly,
    RunYearly,
    RunOnDate,
    RunIfOutOfBounds,
    RunAfterDate,
    RunAfterDays,
    RunEveryNPeriods,
    ClosePositionsAfterDates,
)
from bt.algos.portfolio_ops import Rebalance, RebalanceOverTime, HedgeRisks
from bt.algos.selection import (
    SelectAll,
    SelectRandomly,
    SelectThese,
    SelectWhere,
    SelectRegex,
    SelectHasData,
    SelectN,
    SelectMomentum,
)
from bt.engine import Backtest
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
from bt.algos.flow import Require, Not, Or
from bt.algos.capital import Margin


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


def test_print_temp_data():
    target = mock.MagicMock()
    target.temp = {}
    target.temp["selected"] = ["c1", "c2"]
    target.temp["weights"] = [0.5, 0.5]

    algo = PrintTempData()
    assert algo(target)

    algo = PrintTempData("Selected: {selected}")
    assert algo(target)


def test_print_info():
    target = Strategy("s", [])
    target.temp = {}

    algo = PrintInfo()
    assert algo(target)

    algo = PrintInfo("{now}: {name}")
    assert algo(target)


def test_run_once():
    algo = RunOnce()
    assert algo(None)
    assert not algo(None)
    assert not algo(None)


def test_run_period():
    target = mock.MagicMock()

    dts = pd.date_range("2010-01-01", periods=35)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    algo = RunPeriod()

    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data
    dts = target.data.index

    target.now = None
    assert not algo(target)

    # run on first date
    target.now = dts[0]
    assert not algo(target)

    # run on first supplied date
    target.now = dts[1]
    assert algo(target)

    # run on last date
    target.now = dts[len(dts) - 1]
    assert not algo(target)

    algo = RunPeriod(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )

    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data
    dts = target.data.index

    # run on first date
    target.now = dts[0]
    assert not algo(target)

    # first supplied date
    target.now = dts[1]
    assert not algo(target)

    # run on last date
    target.now = dts[len(dts) - 1]
    assert algo(target)

    # date not in index
    target.now = datetime(2009, 2, 15)
    assert not algo(target)


def test_run_daily():
    target = mock.MagicMock()

    dts = pd.date_range("2010-01-01", periods=35)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    algo = RunDaily()

    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    target.now = dts[1]
    assert algo(target)


def test_run_weekly():
    dts = pd.date_range("2010-01-01", periods=367)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = RunWeekly()
    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    # end of week
    target.now = dts[2]
    assert not algo(target)

    # new week
    target.now = dts[3]
    assert algo(target)

    algo = RunWeekly(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )
    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    # end of week
    target.now = dts[2]
    assert algo(target)

    # new week
    target.now = dts[3]
    assert not algo(target)

    dts = pd.DatetimeIndex(
        [datetime(2016, 1, 3), datetime(2017, 1, 8), datetime(2018, 1, 7)]
    )
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    # check next year
    target.now = dts[1]
    assert algo(target)


def test_run_monthly():
    dts = pd.date_range("2010-01-01", periods=367)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = RunMonthly()
    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    # end of month
    target.now = dts[30]
    assert not algo(target)

    # new month
    target.now = dts[31]
    assert algo(target)

    algo = RunMonthly(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )
    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    # end of month
    target.now = dts[30]
    assert algo(target)

    # new month
    target.now = dts[31]
    assert not algo(target)

    dts = pd.DatetimeIndex(
        [datetime(2016, 1, 3), datetime(2017, 1, 8), datetime(2018, 1, 7)]
    )
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    # check next year
    target.now = dts[1]
    assert algo(target)


def test_run_quarterly():
    dts = pd.date_range("2010-01-01", periods=367)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = RunQuarterly()
    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    # end of quarter
    target.now = dts[89]
    assert not algo(target)

    # new quarter
    target.now = dts[90]
    assert algo(target)

    algo = RunQuarterly(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )
    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    # end of quarter
    target.now = dts[89]
    assert algo(target)

    # new quarter
    target.now = dts[90]
    assert not algo(target)

    dts = pd.DatetimeIndex(
        [datetime(2016, 1, 3), datetime(2017, 1, 8), datetime(2018, 1, 7)]
    )
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    # check next year
    target.now = dts[1]
    assert algo(target)


def test_run_yearly():
    dts = pd.date_range("2010-01-01", periods=367)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    target = mock.MagicMock()
    target.data = data

    algo = RunYearly()
    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    # end of year
    target.now = dts[364]
    assert not algo(target)

    # new year
    target.now = dts[365]
    assert algo(target)

    algo = RunYearly(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )
    # adds the initial day
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    # end of year
    target.now = dts[364]
    assert algo(target)

    # new year
    target.now = dts[365]
    assert not algo(target)


def test_run_on_date():
    target = mock.MagicMock()
    target.now = pd.to_datetime("2010-01-01")

    algo = RunOnDate("2010-01-01", "2010-01-02")
    assert algo(target)

    target.now = pd.to_datetime("2010-01-02")
    assert algo(target)

    target.now = pd.to_datetime("2010-01-03")
    assert not algo(target)


def test_run_if_out_of_bounds():
    algo = RunIfOutOfBounds(0.5)
    dts = pd.date_range("2010-01-01", periods=3)

    s = Strategy("s")
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    s.setup(data)

    s.temp["selected"] = ["c1", "c2"]
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}
    s.update(dts[0])
    s.children["c1"] = SecurityBase("c1")
    s.children["c2"] = SecurityBase("c2")

    s.children["c1"]._weight = 0.5
    s.children["c2"]._weight = 0.5
    assert not algo(s)

    s.children["c1"]._weight = 0.25
    s.children["c2"]._weight = 0.75
    assert not algo(s)

    s.children["c1"]._weight = 0.24
    s.children["c2"]._weight = 0.76
    assert algo(s)

    s.children["c1"]._weight = 0.75
    s.children["c2"]._weight = 0.25
    assert not algo(s)
    s.children["c1"]._weight = 0.76
    s.children["c2"]._weight = 0.24
    assert algo(s)


def test_run_after_date():
    target = mock.MagicMock()
    target.now = pd.to_datetime("2010-01-01")

    algo = RunAfterDate("2010-01-02")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-02")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-03")
    assert algo(target)


def test_run_after_days():
    target = mock.MagicMock()
    target.now = pd.to_datetime("2010-01-01")

    algo = RunAfterDays(3)
    assert not algo(target)
    assert not algo(target)
    assert not algo(target)
    assert algo(target)


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


def test_select_all():
    algo = SelectAll()

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # make sure don't keep nan
    s.update(dts[1])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    # if specify include_no_data then 2
    algo2 = SelectAll(include_no_data=True)

    assert algo2(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    algo3 = SelectAll(include_negative=True)

    assert algo3(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_randomly_n_none():
    algo = SelectRandomly(n=None)  # Behaves like SelectAll

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # make sure don't keep nan
    s.update(dts[1])

    assert algo(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 1
    assert "c2" in selected

    # if specify include_no_data then 2
    algo2 = SelectRandomly(n=None, include_no_data=True)

    assert algo2(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 1
    assert "c2" in selected

    algo3 = SelectRandomly(n=None, include_negative=True)

    assert algo3(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_randomly():

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    data.loc[dts[0], "c1"] = np.nan
    data.loc[dts[0], "c2"] = 95
    data.loc[dts[0], "c3"] = -5

    s.setup(data)
    s.update(dts[0])

    algo = SelectRandomly(n=1)
    assert algo(s)
    assert s.temp.pop("selected") == ["c2"]

    random.seed(1000)
    algo = SelectRandomly(n=1, include_negative=True)
    assert algo(s)
    assert s.temp.pop("selected") == ["c3"]

    random.seed(1009)
    algo = SelectRandomly(n=1, include_no_data=True)
    assert algo(s)
    assert s.temp.pop("selected") == ["c1"]

    random.seed(1009)
    # If selected already set, it will further filter it
    s.temp["selected"] = ["c2"]
    algo = SelectRandomly(n=1, include_no_data=True)
    assert algo(s)
    assert s.temp.pop("selected") == ["c2"]


def test_select_these():
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    s.setup(data)
    s.update(dts[0])

    algo = SelectThese(["c1", "c2"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    algo = SelectThese(["c1"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c1" in selected

    # make sure don't keep nan
    s.update(dts[1])

    algo = SelectThese(["c1", "c2"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    # if specify include_no_data then 2
    algo2 = SelectThese(["c1", "c2"], include_no_data=True)

    assert algo2(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    algo3 = SelectThese(["c1", "c2"], include_negative=True)

    assert algo3(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_where_all():
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    where = pd.DataFrame(index=dts, columns=["c1", "c2"], data=True)

    s.setup(data, where=where)
    s.update(dts[0])

    algo = SelectWhere("where")
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # make sure don't keep nan
    s.update(dts[1])

    algo = SelectThese(["c1", "c2"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    # if specify include_no_data then 2
    algo2 = SelectWhere("where", include_no_data=True)

    assert algo2(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    algo3 = SelectWhere("where", include_negative=True)

    assert algo3(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_where():
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    where = pd.DataFrame(index=dts, columns=["c1", "c2"], data=True)
    where.loc[dts[1]] = False
    where.loc[dts[2], "c1"] = False

    algo = SelectWhere("where")

    s.setup(data, where=where)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    s.update(dts[1])
    assert algo(s)
    assert s.temp["selected"] == []

    s.update(dts[2])
    assert algo(s)
    assert s.temp["selected"] == ["c2"]


def test_select_where_legacy():
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    where = pd.DataFrame(index=dts, columns=["c1", "c2"], data=True)
    where.loc[dts[1]] = False
    where.loc[dts[2], "c1"] = False

    algo = SelectWhere(where)

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    s.update(dts[1])
    assert algo(s)
    assert s.temp["selected"] == []

    s.update(dts[2])
    assert algo(s)
    assert s.temp["selected"] == ["c2"]


def test_select_regex():
    s = Strategy("s")
    algo = SelectRegex("c1")

    s.temp["selected"] = ["a1", "c1", "c2", "c11", "cc1"]
    assert algo(s)
    assert s.temp["selected"] == ["c1", "c11", "cc1"]

    algo = SelectRegex("^c1$")
    assert algo(s)
    assert s.temp["selected"] == ["c1"]


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


def test_select_has_data():
    algo = SelectHasData(min_count=3, lookback=pd.DateOffset(days=3))

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=10)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[0], "c1"] = np.nan
    data.loc[dts[1], "c1"] = np.nan

    s.setup(data)
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected


def test_select_has_data_preselected():
    algo = SelectHasData(min_count=3, lookback=pd.DateOffset(days=3))

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[0], "c1"] = np.nan
    data.loc[dts[1], "c1"] = np.nan

    s.setup(data)
    s.update(dts[2])
    s.temp["selected"] = ["c1"]

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 0


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


def test_select_n():
    algo = SelectN(n=1, sort_descending=True)

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[2], "c1"] = 105
    data.loc[dts[2], "c2"] = 95

    s.setup(data)
    s.update(dts[2])
    first = data.iloc[0]
    last = data.iloc[-1]
    s.temp["stat"] = (last / first) - 1

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c1" in selected

    algo = SelectN(n=1, sort_descending=False, filter_selected=False)
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    # return 2 we have if all_or_none false
    algo = SelectN(n=3, sort_descending=False, filter_selected=False)
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # return 0 we have if all_or_none true
    algo = SelectN(n=3, sort_descending=False, all_or_none=True, filter_selected=False)
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 0


def test_select_n_perc():
    algo = SelectN(n=0.5, sort_descending=True)

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[2], "c1"] = 105
    data.loc[dts[2], "c2"] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp["stat"] = data.calc_total_return()

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c1" in selected


def test_select_momentum():
    algo = SelectMomentum(n=1, lookback=pd.DateOffset(days=3))

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[2], "c1"] = 105
    data.loc[dts[2], "c2"] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    actual = s.temp["selected"]
    assert len(actual) == 1
    assert "c1" in actual


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


def test_require():
    target = mock.MagicMock()
    target.temp = {}

    algo = Require(lambda x: len(x) > 0, "selected")
    assert not algo(target)

    target.temp["selected"] = []
    assert not algo(target)

    target.temp["selected"] = ["a", "b"]
    assert algo(target)


def test_run_every_n_periods():
    target = mock.MagicMock()
    target.temp = {}

    algo = RunEveryNPeriods(n=3, offset=0)

    target.now = pd.to_datetime("2010-01-01")
    assert algo(target)
    # run again w/ no date change should not trigger
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-02")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-03")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-04")
    assert algo(target)

    target.now = pd.to_datetime("2010-01-05")
    assert not algo(target)


def test_run_every_n_periods_offset():
    target = mock.MagicMock()
    target.temp = {}

    algo = RunEveryNPeriods(n=3, offset=1)

    target.now = pd.to_datetime("2010-01-01")
    assert not algo(target)
    # run again w/ no date change should not trigger
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-02")
    assert algo(target)

    target.now = pd.to_datetime("2010-01-03")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-04")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-05")
    assert algo(target)


def test_not():
    target = mock.MagicMock()
    target.temp = {}

    # run except on the 1/2/18
    runOnDateAlgo = RunOnDate(pd.to_datetime("2018-01-02"))
    notAlgo = Not(runOnDateAlgo)

    target.now = pd.to_datetime("2018-01-01")
    assert notAlgo(target)

    target.now = pd.to_datetime("2018-01-02")
    assert not notAlgo(target)


def test_or():
    target = mock.MagicMock()
    target.temp = {}

    # run on the 1/2/18
    runOnDateAlgo = RunOnDate(pd.to_datetime("2018-01-02"))
    runOnDateAlgo2 = RunOnDate(pd.to_datetime("2018-01-03"))
    runOnDateAlgo3 = RunOnDate(pd.to_datetime("2018-01-04"))
    runOnDateAlgo4 = RunOnDate(pd.to_datetime("2018-01-04"))

    orAlgo = Or([runOnDateAlgo, runOnDateAlgo2, runOnDateAlgo3, runOnDateAlgo4])

    # verify it returns false when neither is true
    target.now = pd.to_datetime("2018-01-01")
    assert not orAlgo(target)

    # verify it returns true when the first is true
    target.now = pd.to_datetime("2018-01-02")
    assert orAlgo(target)

    # verify it returns true when the second is true
    target.now = pd.to_datetime("2018-01-03")
    assert orAlgo(target)

    # verify it returns true when both algos return true
    target.now = pd.to_datetime("2018-01-04")
    assert orAlgo(target)


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


def test_margin():
    algo = Margin(0.1, 0.66666666667)

    s = Strategy("s", algos=[WeighSpecified(c1=2), Rebalance()])

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1"], data=1)

    yesterday = dts[0] - timedelta(days=1)
    algo._last_date = yesterday

    s.setup(data)
    s.update(dts[0])
    s.adjust(1000)
    s.run()

    algo(s)

    # checked that we charged some margin interest
    fees = np.sum(s.fees)
    assert pytest.approx(0.26, 0.01) == fees

    # check that we've liquidated things to get us back to the maintenance requirement
    assert pytest.approx(1499, 0.001) == sum(
        child.value for child in s.children.values()
    )

    assert pytest.approx(999.73, 0.001) == s.value
