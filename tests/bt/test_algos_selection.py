import random

import pandas as pd
import pytest

from bt.algos.selection import (
    AddSecurity,
    Ranking,
    RemoveSecurities,
    SectorDoubleSort,
    StandardDoubleSort,
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


class _TestRanking(Ranking):
    def __call__(self, target: Strategy) -> bool:
        resolved = self._resolve_ranking_context(target)
        if resolved is None:
            return False
        temp, _, now, candidate_pool = resolved

        ranked = self._prepare_ranked_stat(temp, "metric", candidate_pool)
        if ranked is None:
            return False
        if ranked.empty:
            return self._set_empty_selection(temp, now)

        selected_names = list(ranked.index)
        return self._set_selected_and_record_stats(temp, now, selected_names, ranked)


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


def test_ranking_base_prepares_ranked_stat_and_records_stats():
    prices = _prices(
        A=[100.0, 101.0], B=[100.0, 101.0], C=[100.0, 101.0], D=[100.0, 101.0]
    )
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B", "C"]
    strategy.temp["metric"] = pd.Series(
        {"A": 2.0, "B": float("nan"), "C": float("inf"), "D": 4.0}
    )

    algo = _TestRanking()
    assert algo(strategy)
    assert strategy.temp["selected"] == ["A"]
    assert algo.stats.loc[prices.index[1], "TOTAL_NAMES"] == 1
    assert algo.stats.loc[prices.index[1], "MEAN"] == 2.0
    assert algo.stats.loc[prices.index[1], "MEDIAN"] == 2.0


def test_ranking_base_sets_empty_selection_and_empty_stats():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 101.0])
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B"]
    strategy.temp["metric"] = pd.Series({"A": float("nan"), "B": float("inf")})

    algo = _TestRanking()
    assert algo(strategy)
    assert strategy.temp["selected"] == []
    assert algo.stats.loc[prices.index[1], "TOTAL_NAMES"] == 0
    assert pd.isna(algo.stats.loc[prices.index[1], "MEAN"])
    assert pd.isna(algo.stats.loc[prices.index[1], "MEDIAN"])


def test_select_n_supports_absolute_and_fractional_counts():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 101.0], C=[100.0, 101.0])
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B", "C"]
    strategy.temp["momentum"] = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})

    assert SelectN(n=2, stat_key="momentum")(strategy)
    assert strategy.temp["selected"] == ["A", "B"]

    strategy.temp["selected"] = ["A", "B", "C"]
    assert SelectN(n=0.5, stat_key="momentum")(strategy)
    assert strategy.temp["selected"] == ["A"]


def test_select_n_requires_explicit_non_empty_stat_key():
    with pytest.raises(TypeError, match="stat_key"):
        SelectN(n=1, stat_key="")


def test_select_quantile_picks_requested_bucket():
    prices = _prices(
        A=[100.0, 101.0], B=[100.0, 101.0], C=[100.0, 101.0], D=[100.0, 101.0]
    )
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B", "C", "D"]
    strategy.temp["momentum"] = pd.Series({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})

    assert SelectQuantile(n_tiles=2, tile=2, stat_key="momentum")(strategy)
    assert strategy.temp["selected"] == ["C", "D"]


def test_select_quantile_requires_explicit_non_empty_stat_key():
    with pytest.raises(TypeError, match="stat_key"):
        SelectQuantile(n_tiles=2, tile=1, stat_key="")


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
    strategy.temp["momentum"] = pd.Series({"A": 4.0, "B": 1.0, "C": 3.0, "D": 2.0})

    assert SectorDoubleSort(n_tiles=2, stat_key="momentum")(strategy)
    assert strategy.temp["selected"] == ["A", "C"]


def test_sector_double_sort_requires_explicit_non_empty_stat_key():
    with pytest.raises(TypeError, match="stat_key"):
        SectorDoubleSort(n_tiles=2, stat_key="")


def test_standard_double_sort_selects_top_second_bucket_within_each_first_bucket():
    prices = _prices(
        A=[100.0, 101.0],
        B=[100.0, 101.0],
        C=[100.0, 101.0],
        D=[100.0, 101.0],
        E=[100.0, 101.0],
        F=[100.0, 101.0],
        G=[100.0, 101.0],
        H=[100.0, 101.0],
    )
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B", "C", "D", "E", "F", "G", "H"]
    strategy.temp["size"] = pd.Series(
        {"A": 8.0, "B": 7.0, "C": 6.0, "D": 5.0, "E": 4.0, "F": 3.0, "G": 2.0, "H": 1.0}
    )
    strategy.temp["quality"] = pd.Series(
        {"A": 1.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 8.0, "F": 5.0, "G": 7.0, "H": 6.0}
    )

    assert StandardDoubleSort(
        n_tiles_1=2,
        n_tiles_2=2,
        stat_key_1="size",
        stat_key_2="quality",
    )(strategy)
    assert strategy.temp["selected"] == ["B", "C", "E", "G"]


def test_standard_double_sort_requires_explicit_non_empty_stat_keys():
    with pytest.raises(TypeError, match="stat_key_1"):
        StandardDoubleSort(
            n_tiles_1=2,
            n_tiles_2=2,
            stat_key_1="",
            stat_key_2="quality",
        )
    with pytest.raises(TypeError, match="stat_key_2"):
        StandardDoubleSort(
            n_tiles_1=2,
            n_tiles_2=2,
            stat_key_1="size",
            stat_key_2="",
        )


def test_standard_double_sort_sort_descending_controls_second_metric_only():
    prices = _prices(
        A=[100.0, 101.0],
        B=[100.0, 101.0],
        C=[100.0, 101.0],
        D=[100.0, 101.0],
        E=[100.0, 101.0],
        F=[100.0, 101.0],
        G=[100.0, 101.0],
        H=[100.0, 101.0],
    )
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B", "C", "D", "E", "F", "G", "H"]
    strategy.temp["size"] = pd.Series(
        {"A": 8.0, "B": 7.0, "C": 6.0, "D": 5.0, "E": 4.0, "F": 3.0, "G": 2.0, "H": 1.0}
    )
    strategy.temp["quality"] = pd.Series(
        {"A": 1.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 8.0, "F": 5.0, "G": 7.0, "H": 6.0}
    )

    assert StandardDoubleSort(
        n_tiles_1=2,
        n_tiles_2=2,
        stat_key_1="size",
        stat_key_2="quality",
        sort_descending=False,
    )(strategy)
    assert strategy.temp["selected"] == ["A", "D", "F", "H"]


def test_standard_double_sort_supports_different_tile_counts_per_pass():
    prices = _prices(
        A=[100.0, 101.0],
        B=[100.0, 101.0],
        C=[100.0, 101.0],
        D=[100.0, 101.0],
        E=[100.0, 101.0],
        F=[100.0, 101.0],
    )
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B", "C", "D", "E", "F"]
    strategy.temp["size"] = pd.Series(
        {"A": 6.0, "B": 5.0, "C": 4.0, "D": 3.0, "E": 2.0, "F": 1.0}
    )
    strategy.temp["quality"] = pd.Series(
        {"A": 1.0, "B": 3.0, "C": 2.0, "D": 6.0, "E": 4.0, "F": 5.0}
    )

    assert StandardDoubleSort(
        n_tiles_1=3,
        n_tiles_2=2,
        stat_key_1="size",
        stat_key_2="quality",
    )(strategy)
    assert strategy.temp["selected"] == ["B", "D", "F"]


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
