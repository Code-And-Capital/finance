import pandas as pd
import pytest

from bt.algos.selection import SelectN
from bt.algos.signals import (
    DualMACrossoverSignal,
    MomentumSignal,
    PriceCrossOverSignal,
)
from bt.core import Strategy


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
) -> Strategy:
    strategy = Strategy("s")
    strategy.setup(prices)
    strategy.now = prices.index[now_idx]
    strategy.last_day = prices.index[now_idx if last_day_idx is None else last_day_idx]
    strategy.inow = now_idx
    strategy.temp = {}
    strategy.perm = {}
    return strategy


def test_price_crossover_signal_uses_last_day_prices():
    prices = _prices(A=[100.0, 110.0, 90.0], B=[100.0, 90.0, 120.0])
    strategy = _strategy_context(prices, now_idx=2, last_day_idx=1)
    strategy.temp["moving_average"] = pd.Series({"A": 105.0, "B": 95.0})
    algo = PriceCrossOverSignal()

    assert algo(strategy)
    assert strategy.temp["selected"] == ["A"]
    assert bool(algo.history.loc[prices.index[1], "A"]) is True
    assert bool(algo.history.loc[prices.index[1], "B"]) is False


def test_price_crossover_signal_respects_existing_candidate_pool():
    prices = _prices(A=[100.0, 110.0], B=[100.0, 90.0])
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["moving_average"] = pd.Series({"A": 105.0, "B": 95.0})
    strategy.temp["selected"] = ["B"]
    algo = PriceCrossOverSignal()

    assert algo(strategy)
    assert strategy.temp["selected"] == []
    assert bool(algo.history.loc[prices.index[1], "A"]) is False
    assert bool(algo.history.loc[prices.index[1], "B"]) is False


def test_price_crossover_signal_returns_false_when_reference_missing():
    prices = _prices(A=[100.0])
    strategy = _strategy_context(prices, now_idx=0)

    assert not PriceCrossOverSignal(ma_name="missing")(strategy)


def test_dual_ma_crossover_signal_selects_short_over_long():
    prices = _prices(A=[100.0], B=[100.0], C=[100.0])
    strategy = _strategy_context(prices, now_idx=0)
    strategy.temp["ma_short"] = pd.Series({"A": 102.0, "B": 98.0, "C": 101.0})
    strategy.temp["ma_long"] = pd.Series({"A": 100.0, "B": 100.0, "C": 101.0})

    assert DualMACrossoverSignal()(strategy)
    assert strategy.temp["selected"] == ["A"]


def test_momentum_signal_validates_inputs():
    with pytest.raises(TypeError, match="ranking_algo"):
        MomentumSignal(ranking_algo=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="total_return_key"):
        MomentumSignal(ranking_algo=SelectN(n=1), total_return_key=123)  # type: ignore[arg-type]


def test_momentum_signal_uses_ranking_algo_to_choose_top_names():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 99.0], C=[100.0, 105.0])
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B", "C"]
    strategy.temp["total_return"] = pd.Series({"A": 0.01, "B": -0.01, "C": 0.05})

    algo = MomentumSignal(
        ranking_algo=SelectN(n=1, sort_descending=True, stat_key="total_return"),
        total_return_key="total_return",
    )
    assert algo(strategy)
    assert strategy.temp["selected"] == ["C"]


def test_momentum_signal_returns_false_without_return_series():
    prices = _prices(A=[100.0, 101.0])
    strategy = _strategy_context(prices, now_idx=1)

    assert not MomentumSignal(ranking_algo=SelectN(n=1))(strategy)
