from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from bt.algos.capital import CapitalFlow, Margin


def test_capital_flow_adjusts_target():
    target = mock.MagicMock()

    assert CapitalFlow(125.5)(target)
    target.adjust.assert_called_once_with(125.5)


def test_capital_flow_requires_adjust_method():
    with pytest.raises(AttributeError, match="adjust"):
        CapitalFlow(100.0)(object())


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


def test_margin_first_call_only_seeds_last_date():
    algo = Margin(0.1, 0.5)
    target = mock.MagicMock()
    target.now = pd.Timestamp("2024-01-01")
    target.capital = -100.0
    target.children = {"A": SimpleNamespace(value=1_000.0)}
    target.value = 900.0

    assert algo(target)
    target.adjust.assert_not_called()
    target.allocate.assert_not_called()


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
    target.children = {"A": SimpleNamespace(value=1_000.0)}
    target.value = 900.0
    target.capital = -100.0
    target.now = pd.Timestamp("2024-01-01")

    assert algo(target)

    target.now = pd.Timestamp("2024-01-02")
    assert algo(target)

    target.adjust.assert_called_once()
    _, kwargs = target.adjust.call_args
    assert kwargs["flow"] is False
    assert kwargs["fee"] > 0.0


def test_margin_triggers_liquidation_when_equity_ratio_breaches_requirement():
    algo = Margin(0.1, 0.5)
    target = mock.MagicMock()
    target.children = {"A": SimpleNamespace(value=1_000.0)}
    target.value = 100.0
    target.capital = -100.0
    target.now = pd.Timestamp("2024-01-01")

    assert algo(target)

    target.now = pd.Timestamp("2024-01-02")
    assert algo(target)

    target.allocate.assert_called_once_with(-400.0)
