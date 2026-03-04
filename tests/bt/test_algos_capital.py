from datetime import timedelta
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from bt.core import Strategy
from bt.algos.capital import CapitalFlow, Margin
from bt.algos.portfolio_ops import Rebalance
from bt.algos.weighting import WeighSpecified


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

    fees = np.sum(s.fees)
    assert pytest.approx(0.26, 0.01) == fees

    assert pytest.approx(1499, 0.001) == sum(
        child.value for child in s.children.values()
    )

    assert pytest.approx(999.73, 0.001) == s.value


def test_capital_flow_adjusts_target_capital():
    target = mock.MagicMock()
    algo = CapitalFlow(125.5)

    assert algo(target)
    target.adjust.assert_called_once_with(125.5)


def test_capital_flow_requires_adjust_method():
    algo = CapitalFlow(100.0)

    with pytest.raises(AttributeError, match="adjust"):
        algo(object())


def test_margin_validates_constructor_inputs():
    with pytest.raises(ValueError, match="greater than -1.0"):
        Margin(-1.0, 0.5)

    with pytest.raises(ValueError, match="range"):
        Margin(0.1, 0.0)

    with pytest.raises(ValueError, match="range"):
        Margin(0.1, 1.5)


def test_margin_requires_target_interface():
    algo = Margin(0.1, 0.5)

    with pytest.raises(AttributeError, match="missing required attributes"):
        algo(object())


def test_margin_raises_on_non_monotonic_dates():
    algo = Margin(0.1, 0.5)
    target = mock.MagicMock()
    target.now = pd.Timestamp("2024-01-10")
    target.capital = 0.0
    target.children = {}
    target.value = 0.0

    assert algo(target)

    target.now = pd.Timestamp("2024-01-09")
    with pytest.raises(ValueError, match="non-monotonic"):
        algo(target)


def test_margin_interest_is_applied_as_non_flow_fee():
    algo = Margin(0.1, 0.5)
    target = mock.MagicMock()
    target.children = {"c1": mock.MagicMock(value=1000.0)}
    target.value = 1000.0
    target.capital = -100.0
    target.now = pd.Timestamp("2024-01-01")

    assert algo(target)

    target.now = pd.Timestamp("2024-01-02")
    assert algo(target)

    _, kwargs = target.adjust.call_args
    assert kwargs["flow"] is False
    assert kwargs["fee"] > 0.0
