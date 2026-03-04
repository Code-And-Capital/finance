from datetime import datetime
from unittest import mock

import pandas as pd
import pytest

from bt.core import Algo, Strategy, SecurityBase
from bt.engine import Backtest
from bt.algos.flow import (
    Require,
    Not,
    Or,
    RunOnce,
    RunPeriod,
    RunDaily,
    RunWeekly,
    RunMonthly,
    RunQuarterly,
    RunYearly,
    RunOnDate,
    RunIfOutOfBounds,
    RunIfCashOutOfBounds,
    RunAfterDate,
    RunAfterDays,
    RunEveryNPeriods,
    ClosePositionsAfterDates,
)


def test_run_once_ignores_inow_zero():
    algo = RunOnce()
    target = mock.MagicMock()
    target.inow = 0
    assert not algo(target)

    target.inow = 1
    assert algo(target)
    assert not algo(target)


@pytest.mark.parametrize(
    "run_on_first_call,expected",
    [
        (True, [True, False, False]),
        (False, [False, False]),
    ],
)
def test_run_once_sequences(run_on_first_call, expected):
    algo = RunOnce(run_on_first_call=run_on_first_call)
    assert [algo(None) for _ in expected] == expected


def test_run_period():
    target = mock.MagicMock()

    dts = pd.date_range("2010-01-01", periods=35)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    algo = RunPeriod()

    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data
    dts = target.data.index

    target.now = None
    assert not algo(target)

    target.now = dts[0]
    assert not algo(target)

    target.now = dts[1]
    assert algo(target)

    target.now = dts[len(dts) - 1]
    assert not algo(target)

    algo = RunPeriod(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )

    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data
    dts = target.data.index

    target.now = dts[0]
    assert not algo(target)

    target.now = dts[1]
    assert not algo(target)

    target.now = dts[len(dts) - 1]
    assert algo(target)

    target.now = datetime(2009, 2, 15)
    assert not algo(target)


def test_run_period_handles_missing_data_and_invalid_now():
    algo = RunDaily()

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100)
    target = mock.MagicMock()
    target.data = data
    target.now = "not-a-date"
    assert not algo(target)


def test_run_daily():
    target = mock.MagicMock()

    dts = pd.date_range("2010-01-01", periods=35)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    algo = RunDaily()

    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data

    target.now = dts[1]
    assert algo(target)


def _setup_period_target(algo, dts):
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)
    target = mock.MagicMock()
    backtest = Backtest(Strategy("", [algo]), data)
    target.data = backtest.data
    return target


@pytest.mark.parametrize(
    "algo_cls,end_idx,next_idx",
    [
        (RunWeekly, 2, 3),
        (RunMonthly, 30, 31),
        (RunQuarterly, 89, 90),
        (RunYearly, 364, 365),
    ],
)
def test_period_frequency_default(algo_cls, end_idx, next_idx):
    dts = pd.date_range("2010-01-01", periods=367)
    algo = algo_cls()
    target = _setup_period_target(algo, dts)

    target.now = dts[end_idx]
    assert not algo(target)

    target.now = dts[next_idx]
    assert algo(target)


@pytest.mark.parametrize(
    "algo_cls,end_idx,next_idx,cross_year_expected",
    [
        (RunWeekly, 2, 3, False),
        (RunMonthly, 30, 31, False),
        (RunQuarterly, 89, 90, False),
        (RunYearly, 364, 365, None),
    ],
)
def test_period_frequency_end_of_period(
    algo_cls, end_idx, next_idx, cross_year_expected
):
    dts = pd.date_range("2010-01-01", periods=367)
    algo = algo_cls(
        run_on_first_date=False, run_on_end_of_period=True, run_on_last_date=True
    )
    target = _setup_period_target(algo, dts)

    target.now = dts[end_idx]
    assert algo(target)

    target.now = dts[next_idx]
    assert not algo(target)

    if cross_year_expected is not None:
        cross_year_dts = pd.DatetimeIndex(
            [datetime(2016, 1, 3), datetime(2017, 1, 8), datetime(2018, 1, 7)]
        )
        target = _setup_period_target(algo, cross_year_dts)
        target.now = cross_year_dts[1]
        assert algo(target) is cross_year_expected


@pytest.mark.parametrize(
    "dates,matching_now,non_matching_now",
    [
        (["2010-01-01", "2010-01-02"], "2010-01-02", "2010-01-03"),
        ("2010-01-02", "2010-01-02", "2010-01-03"),
    ],
)
def test_run_on_date_accepts_scalar_or_iterable(dates, matching_now, non_matching_now):
    target = mock.MagicMock()
    algo = RunOnDate(dates)

    target.now = pd.to_datetime(matching_now)
    assert algo(target)

    target.now = pd.to_datetime(non_matching_now)
    assert not algo(target)


def test_run_on_date_rejects_invalid_configured_dates():
    with pytest.raises(ValueError, match="valid timestamp"):
        RunOnDate([pd.NaT])


def test_run_on_date_handles_missing_or_invalid_now():
    algo = RunOnDate(["2010-01-01"])

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    target.now = "not-a-date"
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


def test_run_if_out_of_bounds_validates_tolerance():
    with pytest.raises(ValueError, match=">= 0"):
        RunIfOutOfBounds(-0.1)


def test_run_if_out_of_bounds_validates_mode():
    with pytest.raises(ValueError, match="mode"):
        RunIfOutOfBounds(0.1, mode="bad")


def test_run_if_out_of_bounds_handles_zero_target_weight():
    algo = RunIfOutOfBounds(0.10)
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100)
    s.setup(data)
    s.update(dts[0])
    s.children["c1"] = SecurityBase("c1")
    s.children["c1"]._weight = 0.20
    s.temp["weights"] = {"c1": 0.0}

    assert algo(s)


def test_run_if_cash_out_of_bounds():
    algo = RunIfCashOutOfBounds(0.05)
    target = mock.MagicMock()
    target.temp = {"weights": {"c1": 1.0}, "cash": 0.10}
    target.children = {"c1": mock.MagicMock(weight=1.0)}
    target.capital = 100.0
    target.value = 1000.0
    assert not algo(target)

    target.capital = 200.0
    target.value = 1000.0
    assert algo(target)


def test_run_if_cash_out_of_bounds_validation_and_missing_cash():
    with pytest.raises(ValueError, match=">= 0"):
        RunIfCashOutOfBounds(-0.01)

    algo = RunIfCashOutOfBounds(0.01)
    target = mock.MagicMock()
    target.temp = {"weights": {"c1": 1.0}}
    target.capital = 100.0
    target.value = 1000.0
    assert not algo(target)


def test_run_if_out_of_bounds_accepts_series_weights():
    algo = RunIfOutOfBounds(0.11)
    target = mock.MagicMock()
    target.temp = {"weights": pd.Series({"c1": 0.5, "c2": 0.5})}
    target.children = {
        "c1": mock.MagicMock(weight=0.55),
        "c2": mock.MagicMock(weight=0.45),
    }
    assert not algo(target)


def test_run_if_out_of_bounds_absolute_mode():
    algo = RunIfOutOfBounds(0.10, mode="absolute")
    target = mock.MagicMock()
    target.temp = {"weights": {"c1": 0.50}}
    target.children = {"c1": mock.MagicMock(weight=0.58)}
    assert not algo(target)

    target.children["c1"].weight = 0.62
    assert algo(target)


def test_run_if_out_of_bounds_ignores_cash_key():
    algo = RunIfOutOfBounds(0.01)
    target = mock.MagicMock()
    target.temp = {"weights": {"c1": 1.0, "cash": 0.0}}
    target.children = {"c1": mock.MagicMock(weight=1.0)}
    target.capital = 1000.0
    target.value = 1000.0
    assert not algo(target)


def test_run_if_out_of_bounds_returns_false_when_state_missing():
    algo = RunIfOutOfBounds(0.1)

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = {}
    assert not algo(target)

    target.temp = {"weights": object()}
    assert not algo(target)


def test_run_if_cash_out_of_bounds_returns_false_when_state_missing():
    algo = RunIfCashOutOfBounds(0.1)

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = {"cash": 0.1}
    target.value = 0.0
    target.capital = 0.0
    assert not algo(target)


@pytest.mark.parametrize(
    "now_value",
    [None, "not-a-date"],
)
def test_run_after_date_handles_missing_or_invalid_now(now_value):
    algo = RunAfterDate("2010-01-02")

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    target.now = now_value
    assert not algo(target)


def test_run_after_date():
    target = mock.MagicMock()
    target.now = pd.to_datetime("2010-01-01")

    algo = RunAfterDate("2010-01-02")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-02")
    assert not algo(target)

    target.now = pd.to_datetime("2010-01-03")
    assert algo(target)


def test_run_after_date_rejects_nat():
    with pytest.raises(ValueError, match="valid timestamp"):
        RunAfterDate(pd.NaT)


@pytest.mark.parametrize(
    "days,expected",
    [
        (3, [False, False, False, True]),
        (0, [True, True]),
    ],
)
def test_run_after_days_sequences(days, expected):
    algo = RunAfterDays(days)
    assert [algo(None) for _ in expected] == expected


def test_run_after_days_validates_input():
    with pytest.raises(TypeError, match="integer"):
        RunAfterDays(1.5)

    with pytest.raises(ValueError, match=">= 0"):
        RunAfterDays(-1)


def test_require():
    target = mock.MagicMock()
    target.temp = {}

    algo = Require(lambda x: len(x) > 0, "selected")
    assert not algo(target)

    target.temp["selected"] = []
    assert not algo(target)

    target.temp["selected"] = ["a", "b"]
    assert algo(target)


def test_require_validates_predicate_callable():
    with pytest.raises(TypeError, match="callable"):
        Require(pred=object(), item="selected")


def test_require_handles_missing_or_invalid_temp():
    algo = Require(lambda x: len(x) > 0, "selected", if_none=False)
    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = []
    assert not algo(target)


def test_require_if_none_behavior():
    algo = Require(lambda x: len(x) > 0, "selected", if_none=True)
    target = mock.MagicMock()
    target.temp = {}
    assert algo(target)


def test_not():
    target = mock.MagicMock()
    target.temp = {}

    runOnDateAlgo = RunOnDate([pd.to_datetime("2018-01-02")])
    notAlgo = Not(runOnDateAlgo)

    target.now = pd.to_datetime("2018-01-01")
    assert notAlgo(target)

    target.now = pd.to_datetime("2018-01-02")
    assert not notAlgo(target)


def test_or():
    target = mock.MagicMock()
    target.temp = {}

    runOnDateAlgo = RunOnDate([pd.to_datetime("2018-01-02")])
    runOnDateAlgo2 = RunOnDate([pd.to_datetime("2018-01-03")])
    runOnDateAlgo3 = RunOnDate([pd.to_datetime("2018-01-04")])
    runOnDateAlgo4 = RunOnDate([pd.to_datetime("2018-01-04")])

    orAlgo = Or([runOnDateAlgo, runOnDateAlgo2, runOnDateAlgo3, runOnDateAlgo4])

    target.now = pd.to_datetime("2018-01-01")
    assert not orAlgo(target)

    target.now = pd.to_datetime("2018-01-02")
    assert orAlgo(target)

    target.now = pd.to_datetime("2018-01-03")
    assert orAlgo(target)

    target.now = pd.to_datetime("2018-01-04")
    assert orAlgo(target)


def test_not_validates_callable():
    with pytest.raises(TypeError, match="callable"):
        Not(object())


def test_or_validates_iterable_and_callables():
    with pytest.raises(TypeError, match="iterable"):
        Or(None)

    with pytest.raises(TypeError, match="callable"):
        Or([RunOnDate(["2018-01-02"]), object()])


class DummyAlgo(Algo):
    def __init__(self, return_value=True):
        self.return_value = return_value
        self.called = False

    def __call__(self, target):
        self.called = True
        return self.return_value


def test_or_short_circuits_after_first_true():
    target = mock.MagicMock()
    first = DummyAlgo(return_value=False)
    second = DummyAlgo(return_value=True)
    third = DummyAlgo(return_value=True)

    algo = Or([first, second, third])
    assert algo(target)
    assert first.called
    assert second.called
    assert not third.called


@pytest.mark.parametrize(
    "offset,expected",
    [
        (0, [True, False, False, False, True, False]),
        (1, [False, False, True, False, False, True]),
    ],
)
def test_run_every_n_periods_sequences(offset, expected):
    target = mock.MagicMock()
    target.temp = {}

    algo = RunEveryNPeriods(n=3, offset=offset)
    dts = pd.date_range("2010-01-01", periods=5)
    actual = []
    actual.append(algo(target := mock.MagicMock(now=dts[0], temp={})))
    actual.append(algo(target))
    for dt in dts[1:]:
        target.now = dt
        actual.append(algo(target))
    assert actual == expected


def test_run_every_n_periods_handles_missing_now():
    algo = RunEveryNPeriods(n=3)
    target = mock.MagicMock(spec=[])
    assert not algo(target)


def test_run_every_n_periods_validates_input():
    with pytest.raises(TypeError, match="`n` must be an integer"):
        RunEveryNPeriods(n=1.5)

    with pytest.raises(TypeError, match="`offset` must be an integer"):
        RunEveryNPeriods(n=3, offset=1.5)

    with pytest.raises(ValueError, match="`n` must be > 0"):
        RunEveryNPeriods(n=0)

    with pytest.raises(ValueError, match="`offset` must be >= 0"):
        RunEveryNPeriods(n=3, offset=-1)

    with pytest.raises(ValueError, match="0 <= offset < n"):
        RunEveryNPeriods(n=3, offset=3)


def test_close_positions_after_dates_resolves_data_key_once():
    algo = ClosePositionsAfterDates("cutoffs")
    target = mock.MagicMock()
    target.perm = {}
    target.now = pd.Timestamp("2024-01-10")
    target.children = {"c1": SecurityBase("c1")}
    target.root = mock.MagicMock()
    target.get_data.return_value = pd.Series(
        [pd.Timestamp("2024-01-01")], index=["c1"], name="date"
    )

    assert algo(target)
    assert algo(target)
    assert target.get_data.call_count == 1


def test_close_positions_after_dates_updates_candidate_cache_on_child_changes():
    algo = ClosePositionsAfterDates(
        pd.Series([pd.Timestamp("2024-01-01")], index=["c1"])
    )
    target = mock.MagicMock()
    target.perm = {}
    target.now = pd.Timestamp("2024-01-10")
    target.root = mock.MagicMock()
    target.children = {"c1": SecurityBase("c1")}

    assert algo(target)
    assert "c1" in target.perm["closed"]

    target.children = {"c1": SecurityBase("c1"), "c2": SecurityBase("c2")}
    assert algo(target)
    # c2 is not in close dates, so cache refresh should still avoid closing c2
    assert "c2" not in target.perm["closed"]
