from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from bt.algos.weighting import (
    LimitBenchmarkDeviation,
    LimitDeltas,
    LimitWeights,
    ScaleWeights,
    WeightCurrent,
    WeightEqually,
    WeightFixed,
    WeightFixedSchedule,
    WeightMarket,
    WeightRandomly,
)
from bt.algos.weighting.core import WeightAlgo
from bt.core import Strategy


class DummyWeightAlgo(WeightAlgo):
    def __call__(self, target):
        return True


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
    **kwargs,
) -> Strategy:
    strategy = Strategy("s")
    strategy.setup(prices, **kwargs)
    strategy.now = prices.index[now_idx]
    strategy.last_day = prices.index[now_idx if last_day_idx is None else last_day_idx]
    strategy.inow = now_idx
    strategy.temp = {}
    strategy.perm = {}
    return strategy


def test_weight_algo_normalizes_series_and_records_history():
    algo = DummyWeightAlgo(track_allocation_history=True)
    weights = pd.Series({"A": 0.25, "B": float("nan"), "C": 0.75})
    now = pd.Timestamp("2024-01-02")
    temp = {}

    out = algo._write_weights(temp, weights, now=now, record_history=True)

    assert out == {"A": 0.25, "C": 0.75}
    assert temp["weights"] == {"A": 0.25, "C": 0.75}
    assert algo.allocation_history.loc[now, "A"] == pytest.approx(0.25)


def test_weight_equally_assigns_equal_weights():
    prices = _prices(A=[100.0], B=[100.0], C=[100.0])
    strategy = _strategy_context(prices, now_idx=0)
    strategy.temp["selected"] = ["A", "A", "B"]

    assert WeightEqually()(strategy)
    assert strategy.temp["weights"] == {"A": 0.5, "B": 0.5}


def test_weight_current_bootstraps_then_uses_live_child_weights():
    first_algo = mock.Mock(
        side_effect=lambda target: target.temp.__setitem__("weights", {"A": 1.0})
        or True
    )
    algo = WeightCurrent(first_weight_algo=first_algo)
    target = SimpleNamespace(
        temp={"selected": ["A", "B"]},
        now=pd.Timestamp("2024-01-01"),
        children={},
    )

    assert algo(target)
    assert target.temp["weights"] == {"A": 1.0}

    target.now = pd.Timestamp("2024-01-02")
    target.children = {
        "A": SimpleNamespace(weight=0.75),
        "B": SimpleNamespace(weight=0.25),
    }
    assert algo(target)
    assert target.temp["weights"] == {"A": 0.75, "B": 0.25}


def test_weight_fixed_filters_to_selected_and_normalizes():
    prices = _prices(A=[100.0], B=[100.0], C=[100.0])
    strategy = _strategy_context(prices, now_idx=0)
    strategy.temp["selected"] = ["A", "C"]

    assert WeightFixed(A=2.0, B=1.0, C=1.0)(strategy)
    assert strategy.temp["weights"] == {
        "A": pytest.approx(2 / 3),
        "C": pytest.approx(1 / 3),
    }


def test_scale_weights_scales_existing_mapping():
    target = SimpleNamespace(
        temp={"weights": {"A": 0.2, "B": 0.8}}, now=pd.Timestamp("2024-01-01")
    )

    assert ScaleWeights(0.5)(target)
    assert target.temp["weights"] == {"A": 0.1, "B": 0.4}


def test_weight_fixed_schedule_reads_execution_date_row():
    prices = _prices(A=[100.0, 101.0, 102.0], B=[100.0, 101.0, 102.0])
    schedule = pd.DataFrame(
        {"A": [1.0, 0.25, 0.75], "B": [0.0, 0.75, 0.25]},
        index=prices.index,
    )
    strategy = _strategy_context(prices, now_idx=2, weights_schedule=schedule)

    assert WeightFixedSchedule("weights_schedule")(strategy)
    assert strategy.temp["weights"] == {"A": 0.75, "B": 0.25}


def test_weight_market_reads_market_caps_at_last_day():
    prices = _prices(A=[100.0, 101.0, 102.0], B=[100.0, 101.0, 102.0])
    market_caps = pd.DataFrame(
        {"A": [10.0, 20.0, 30.0], "B": [30.0, 20.0, 10.0]},
        index=prices.index,
    )
    strategy = _strategy_context(
        prices,
        now_idx=2,
        last_day_idx=1,
        marketcap_wide=market_caps,
    )
    strategy.temp["selected"] = ["A", "B"]

    assert WeightMarket()(strategy)
    assert strategy.temp["weights"] == {"A": 0.5, "B": 0.5}


def test_weight_randomly_handles_empty_and_singleton_selection():
    prices = _prices(A=[100.0], B=[100.0])
    strategy = _strategy_context(prices, now_idx=0)

    strategy.temp["selected"] = []
    assert WeightRandomly(random_seed=1)(strategy)
    assert strategy.temp["weights"] == {}

    strategy.temp["selected"] = ["A"]
    assert WeightRandomly(random_seed=1)(strategy)
    assert strategy.temp["weights"] == {"A": 1.0}


def test_limit_weights_caps_and_redistributes_excess():
    target = SimpleNamespace(
        temp={"weights": {"A": 0.8, "B": 0.2}}, now=pd.Timestamp("2024-01-01")
    )

    assert LimitWeights(0.6)(target)
    assert target.temp["weights"]["A"] == pytest.approx(0.6)
    assert target.temp["weights"]["B"] == pytest.approx(0.4)


def test_limit_deltas_clips_against_current_child_weights():
    target = SimpleNamespace(
        temp={"weights": {"A": 0.9, "B": 0.1}},
        now=pd.Timestamp("2024-01-01"),
        children={
            "A": SimpleNamespace(weight=0.5),
            "B": SimpleNamespace(weight=0.5),
        },
    )

    assert LimitDeltas(0.1)(target)
    assert target.temp["weights"]["A"] == pytest.approx(0.6)
    assert target.temp["weights"]["B"] == pytest.approx(0.4)


def test_limit_benchmark_deviation_uses_last_day_benchmark_row():
    prices = _prices(A=[100.0, 101.0, 102.0], B=[100.0, 101.0, 102.0])
    benchmark = pd.DataFrame(
        {"A": [0.2, 0.7, 0.5], "B": [0.8, 0.3, 0.5]},
        index=prices.index,
    )
    strategy = _strategy_context(
        prices,
        now_idx=2,
        last_day_idx=1,
        benchmark_wide=benchmark,
    )
    strategy.temp["selected"] = ["A", "B"]
    strategy.temp["weights"] = {"A": 1.0, "B": 0.0}

    assert LimitBenchmarkDeviation(0.1, "benchmark_wide")(strategy)
    assert strategy.temp["weights"]["A"] == pytest.approx(0.8)
    assert strategy.temp["weights"]["B"] == pytest.approx(0.2)
