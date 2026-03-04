from unittest import mock

import pandas as pd

from bt.core import Strategy
from bt.algos.debugging import Debug, PrintDate, PrintTempData, PrintInfo


def test_print_temp_data():
    target = mock.MagicMock()
    target.temp = {}
    target.temp["selected"] = ["c1", "c2"]
    target.temp["weights"] = [0.5, 0.5]

    algo = PrintTempData()
    assert algo(target)

    algo = PrintTempData("Selected: {selected}")
    assert algo(target)


def test_print_temp_data_uses_logger():
    target = mock.MagicMock()
    target.temp = {"selected": ["c1", "c2"], "weights": [0.5, 0.5]}

    with mock.patch("utils.logging.log") as mock_log:
        algo = PrintTempData()
        assert algo(target)

    mock_log.assert_called_once_with(
        "{'selected': ['c1', 'c2'], 'weights': [0.5, 0.5]}"
    )


def test_print_temp_data_format_error_uses_logger():
    target = mock.MagicMock()
    target.temp = {"selected": ["c1", "c2"]}

    with mock.patch("utils.logging.log") as mock_log:
        algo = PrintTempData("Weights: {weights}")
        assert algo(target)

    mock_log.assert_called_once_with(
        "[PrintTempData] Missing key in target.temp: 'weights'"
    )


import pytest


@pytest.mark.parametrize(
    "prefix,expected",
    [
        (None, "2024-01-15 00:00:00"),
        ("[DATE]", "[DATE] 2024-01-15 00:00:00"),
    ],
)
def test_print_date_uses_logger(prefix, expected):
    target = mock.MagicMock()
    target.now = pd.Timestamp("2024-01-15")

    algo = PrintDate(prefix=prefix)
    with mock.patch("utils.logging.log") as mock_log:
        assert algo(target)

    mock_log.assert_called_once_with(expected)


def test_print_date_missing_now_logs_warning():
    target = mock.MagicMock(spec=[])

    algo = PrintDate()
    with mock.patch("utils.logging.log") as mock_log:
        assert algo(target)

    mock_log.assert_called_once_with(
        "[PrintDate] target is missing required attribute: 'now'"
    )


def test_print_info():
    target = Strategy("s", [])
    target.temp = {}

    algo = PrintInfo()
    assert algo(target)

    algo = PrintInfo("{now}: {name}")
    assert algo(target)


def test_print_info_uses_logger():
    target = Strategy("s", [])
    target.temp = {}
    target.now = pd.Timestamp("2024-01-15")

    with mock.patch("utils.logging.log") as mock_log:
        algo = PrintInfo("{now}: {name}")
        assert algo(target)

    mock_log.assert_called_once_with("2024-01-15 00:00:00: s")


def test_print_temp_data_missing_temp_logs_warning():
    target = mock.MagicMock(spec=[])
    algo = PrintTempData()

    with mock.patch("utils.logging.log") as mock_log:
        assert algo(target)

    mock_log.assert_called_once_with(
        "[PrintTempData] target is missing required attribute: 'temp'"
    )


@pytest.mark.parametrize(
    "condition,expected_calls",
    [
        (lambda _: True, 1),
        (lambda _: False, 0),
    ],
)
def test_debug_condition_controls_breakpoint(condition, expected_calls):
    target = mock.MagicMock()
    algo = Debug(condition=condition)

    with mock.patch("pdb.set_trace") as mock_set_trace:
        assert algo(target)

    assert mock_set_trace.call_count == expected_calls


def test_debug_logs_condition_exception_and_continues():
    target = mock.MagicMock()

    def _raise(_):
        raise RuntimeError("bad condition")

    algo = Debug(condition=_raise)

    with mock.patch("pdb.set_trace") as mock_set_trace:
        with mock.patch("utils.logging.log") as mock_log:
            assert algo(target)

    mock_set_trace.assert_not_called()
    mock_log.assert_called_once()
