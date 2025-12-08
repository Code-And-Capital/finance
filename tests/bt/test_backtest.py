from __future__ import division

import sys
import os

import numpy as np
import pandas as pd
import pytest
import random

from unittest import mock
from bt.engine import Backtest
from bt.core.strategy import Strategy
from bt.algos.selection import SelectAll
from bt.algos.weighting import WeighEqually, WeighSpecified, WeighTarget
from bt.algos.portfolio_ops import Rebalance
from bt.algos.flow import RunDaily
import bt


def test_backtest_copies_strategy():
    s = mock.MagicMock()
    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )

    actual = Backtest(s, data, progress_bar=False)

    assert id(s) != id(actual.strategy)


def test_backtest_dates_set():
    s = mock.MagicMock()
    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )

    actual = Backtest(s, data, progress_bar=False)

    # must account for 't0' addition
    assert len(actual.dates) == len(data.index) + 1
    assert actual.dates[1] == data.index[0]
    assert actual.dates[-1] == data.index[-1]


def test_backtest_auto_name():
    s = mock.MagicMock()
    s.name = "s"
    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )

    actual = Backtest(s, data, progress_bar=False)

    assert actual.name == "s"


def test_initial_capital_set():
    s = mock.MagicMock()
    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )
    s.prices = pd.Series(index=pd.date_range("2010-01-01", periods=5), data=100)

    actual = Backtest(s, data, initial_capital=302, progress_bar=False)
    actual.run()

    s = actual.strategy

    s.adjust.assert_called_with(302)


def test_run_loop():
    s = mock.MagicMock()
    # run loop checks on this
    s.bankrupt = False
    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )
    s.prices = pd.Series(index=pd.date_range("2010-01-01", periods=5), data=100)

    actual = Backtest(s, data, initial_capital=302, progress_bar=False)
    actual.run()

    s = actual.strategy

    # account for first update call to 'setup' initial state
    assert s.update.call_count == 10 + 1
    assert s.run.call_count == 5


def test_turnover():
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["a", "b"], data=100)

    data.loc[dts[1], "a"] = 105
    data.loc[dts[1], "b"] = 95

    data.loc[dts[2], "a"] = 110
    data.loc[dts[2], "b"] = 90

    data.loc[dts[3], "a"] = 115
    data.loc[dts[3], "b"] = 85

    s = Strategy("s", [SelectAll(), WeighEqually(), Rebalance()])

    t = Backtest(s, data, commissions=lambda x, y: 0, progress_bar=False)
    res = bt.run(t)

    t = res.backtests["s"]

    # these numbers were (tediously) calculated in excel
    assert np.allclose(t.turnover[dts[0]], 0.0 / 1000000)
    assert np.allclose(t.turnover[dts[1]], 24985.0 / 1000000)
    assert np.allclose(t.turnover[dts[2]], 24970.0 / 997490)
    assert np.allclose(t.turnover[dts[3]], 25160.0 / 992455)
    assert np.allclose(t.turnover[dts[4]], 76100.0 / 1015285)


def test_can_disable_progress_bar_from_run():
    from contextlib import redirect_stderr
    from io import StringIO

    # Create an in-memory buffer
    output_capture = StringIO()

    data = pd.DataFrame(
        index=pd.date_range("2010-01-01", periods=5), columns=["a", "b"], data=100
    )
    s = Strategy("test", [SelectAll(), WeighEqually(), Rebalance()])

    b = Backtest(s, data)

    # Redirect stderr to the buffer
    with redirect_stderr(output_capture):
        result = bt.run(b, progress_bar=False)

    # confirm that the output is empty
    assert output_capture.getvalue() is ""
    # confirm that we actually ran something
    assert len(result.get_transactions()) > 0


def test_Results_helper_functions():

    names = ["foo", "bar"]
    dates = pd.date_range(
        start="2017-01-01", end="2017-12-31", freq=pd.tseries.offsets.BDay()
    )
    n = len(dates)
    rdf = pd.DataFrame(np.zeros((n, len(names))), index=dates, columns=names)

    np.random.seed(1)
    rdf[names[0]] = np.random.normal(loc=0.1 / n, scale=0.2 / np.sqrt(n), size=n)
    rdf[names[1]] = np.random.normal(loc=0.04 / n, scale=0.05 / np.sqrt(n), size=n)

    pdf = 100 * np.cumprod(1 + rdf)

    # algo to fire on the beginning of every month and to run on the first date
    runDailyAlgo = RunDaily(run_on_first_date=True)

    # algo to set the weights
    #  it will only run when runMonthlyAlgo returns true
    #  which only happens on the first of every month
    weights = pd.Series([0.6, 0.4], index=rdf.columns)
    weighSpecifiedAlgo = WeighSpecified(**weights)

    # algo to rebalance the current weights to weights set by weighSpecified
    #  will only run when weighSpecifiedAlgo returns true
    #  which happens every time it runs
    rebalAlgo = Rebalance()

    # a strategy that rebalances monthly to specified weights
    strat = Strategy("static", [runDailyAlgo, weighSpecifiedAlgo, rebalAlgo])

    backtest = Backtest(strat, pdf, integer_positions=False, progress_bar=False)

    res = bt.run(backtest)

    assert type(res.get_security_weights()) is pd.DataFrame

    assert type(res.get_transactions()) is pd.DataFrame

    assert type(res.get_weights()) is pd.DataFrame


def test_30_min_data():
    names = ["foo"]
    dates = pd.date_range(start="2017-01-01", end="2017-12-31", freq="30min")
    n = len(dates)
    rdf = pd.DataFrame(np.zeros((n, len(names))), index=dates, columns=names)

    np.random.seed(1)
    rdf[names[0]] = np.random.normal(loc=0.1 / n, scale=0.2 / np.sqrt(n), size=n)

    pdf = 100 * np.cumprod(1 + rdf)

    sma50 = pdf.rolling(50).mean()
    sma200 = pdf.rolling(200).mean()

    tw = sma200.copy()
    tw[sma50 > sma200] = 1.0
    tw[sma50 <= sma200] = -1.0
    tw[sma200.isnull()] = 0.0

    ma_cross = Strategy("ma_cross", [WeighTarget(tw), Rebalance()])
    t = Backtest(ma_cross, pdf, progress_bar=False)
    res = bt.run(t)

    wait = 1


def test_additional_data_boolean_dtype_no_warning():
    """Test that boolean dtype in additional_data doesn't raise FutureWarning."""
    import warnings

    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["a", "b"], data=100.0)

    # Create additional data with boolean dtype
    signal = pd.DataFrame(
        index=dts, columns=["signal"], data=[True, False, True, False, True]
    )

    s = Strategy("test", [SelectAll(), WeighEqually(), Rebalance()])

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = Backtest(s, data, additional_data={"signal": signal}, progress_bar=False)
        t.run()

        # Check no FutureWarning about bool-dtype concatenation
        future_warnings = [
            warning
            for warning in w
            if issubclass(warning.category, FutureWarning)
            and "bool-dtype" in str(warning.message).lower()
        ]
        assert len(future_warnings) == 0
