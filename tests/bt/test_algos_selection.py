import random

import pandas as pd
import pytest

from bt.algos.selection import (
    AddSecurity,
    RemoveSecurities,
    SectorDoubleSort,
    SelectActive,
    SelectAll,
    SelectHasData,
    SelectIsOpen,
    SelectN,
    SelectQuantile,
    SelectRandomly,
    SelectSector,
    SelectThese,
    SelectWhere,
)
from bt.core import Security, Strategy


def _prices(**columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        columns,
        index=pd.date_range(
            "2024-01-01", periods=len(next(iter(columns.values()))), freq="D"
        ),
        dtype=float,
    )


def _strategy_context(
    prices: pd.DataFrame,
    *,
    now_idx: int,
    last_day_idx: int | None = None,
    children=None,
    **kwargs,
) -> Strategy:
    strategy = Strategy("s", children=children)
    strategy.setup(prices, **kwargs)
    strategy.now = prices.index[now_idx]
    strategy.last_day = prices.index[now_idx if last_day_idx is None else last_day_idx]
    strategy.inow = now_idx
    strategy.temp = {}
    strategy.perm = {}
    return strategy


def test_select_all_uses_last_day_and_filters_missing_and_negative_prices():
    prices = _prices(A=[100.0, float("nan"), 101.0], B=[100.0, -1.0, 102.0])
    strategy = _strategy_context(prices, now_idx=2, last_day_idx=1)

    assert SelectAll()(strategy)
    assert strategy.temp["selected"] == []

    assert SelectAll(include_no_data=True)(strategy)
    assert strategy.temp["selected"] == ["A", "B"]

    assert SelectAll(include_negative=True)(strategy)
    assert strategy.temp["selected"] == ["B"]


def test_select_has_data_requires_complete_history():
    prices = _prices(A=[100.0, 101.0, 102.0], B=[100.0, float("nan"), 102.0])
    strategy = _strategy_context(prices, now_idx=2)

    assert SelectHasData(lookback=pd.DateOffset(days=2))(strategy)
    assert strategy.temp["selected"] == ["A"]


def test_select_active_removes_closed_and_rolled_names():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 101.0], C=[100.0, 101.0])
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B", "C"]
    strategy.perm["rolled"] = {"B"}
    strategy.perm["closed"] = {"C"}

    assert SelectActive()(strategy)
    assert strategy.temp["selected"] == ["A"]


def test_select_is_open_filters_to_non_zero_child_weights():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 101.0])
    strategy = _strategy_context(
        prices,
        now_idx=1,
        children=[Security("A"), Security("B")],
    )
    strategy.children["A"]._weight = 0.5
    strategy.children["B"]._weight = 0.0

    assert SelectIsOpen()(strategy)
    assert strategy.temp["selected"] == ["A"]


def test_select_these_filters_to_configured_tickers():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 101.0], C=[100.0, 101.0])
    strategy = _strategy_context(prices, now_idx=1)

    assert SelectThese(["B", "C"])(strategy)
    assert strategy.temp["selected"] == ["B", "C"]


def test_select_where_uses_last_day_signal_row():
    prices = _prices(A=[100.0, 101.0, 102.0], B=[100.0, 101.0, 102.0])
    signal = pd.DataFrame(
        {"A": [False, True, False], "B": [True, False, True]},
        index=prices.index,
    )
    strategy = _strategy_context(
        prices,
        now_idx=2,
        last_day_idx=1,
        signal_wide=signal,
    )

    assert SelectWhere("signal_wide")(strategy)
    assert strategy.temp["selected"] == ["A"]


def test_select_n_supports_absolute_and_fractional_counts():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 101.0], C=[100.0, 101.0])
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B", "C"]
    strategy.temp["stat"] = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})

    assert SelectN(n=2)(strategy)
    assert strategy.temp["selected"] == ["A", "B"]

    strategy.temp["selected"] = ["A", "B", "C"]
    assert SelectN(n=0.5)(strategy)
    assert strategy.temp["selected"] == ["A"]


def test_select_quantile_picks_requested_bucket():
    prices = _prices(
        A=[100.0, 101.0], B=[100.0, 101.0], C=[100.0, 101.0], D=[100.0, 101.0]
    )
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B", "C", "D"]
    strategy.temp["stat"] = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})

    assert SelectQuantile(n_tiles=2, tile=2)(strategy)
    assert strategy.temp["selected"] == ["C", "D"]


def test_sector_double_sort_selects_top_bucket_within_each_sector():
    prices = _prices(
        A=[100.0, 101.0], B=[100.0, 101.0], C=[100.0, 101.0], D=[100.0, 101.0]
    )
    sectors = pd.DataFrame(
        {
            "A": ["Tech", "Tech"],
            "B": ["Tech", "Tech"],
            "C": ["Health", "Health"],
            "D": ["Health", "Health"],
        },
        index=prices.index,
    )
    strategy = _strategy_context(prices, now_idx=1, sector_wide=sectors)
    strategy.temp["selected"] = ["A", "B", "C", "D"]
    strategy.temp["stat"] = pd.Series({"A": 4.0, "B": 1.0, "C": 3.0, "D": 2.0})

    assert SectorDoubleSort(n_tiles=2)(strategy)
    assert strategy.temp["selected"] == ["A", "C"]


def test_select_randomly_respects_n_and_existing_pool():
    prices = _prices(A=[100.0], B=[100.0], C=[100.0])
    strategy = _strategy_context(prices, now_idx=0)
    strategy.temp["selected"] = ["A", "C"]

    random.seed(7)
    assert SelectRandomly(n=1)(strategy)
    assert strategy.temp["selected"] in [["A"], ["C"]]


def test_remove_and_add_security_modify_selection_with_closed_cache():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 101.0], C=[100.0, -1.0])
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B"]

    assert RemoveSecurities(["B"])(strategy)
    assert strategy.temp["selected"] == ["A"]

    strategy.perm["closed"] = {"B"}
    assert AddSecurity(["B", "C"])(strategy)
    assert strategy.temp["selected"] == ["A"]


def test_select_sector_filters_by_sector_label():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 101.0], C=[100.0, 101.0])
    sectors = pd.DataFrame(
        {"A": ["Tech", "Tech"], "B": ["Health", "Health"], "C": ["Tech", "Tech"]},
        index=prices.index,
    )
    strategy = _strategy_context(prices, now_idx=1, sector_wide=sectors)
    strategy.temp["selected"] = ["A", "B", "C"]

    assert SelectSector(["Tech"])(strategy)
    assert strategy.temp["selected"] == ["A", "C"]
