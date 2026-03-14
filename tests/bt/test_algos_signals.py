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
    algo = PriceCrossOverSignal(
        ma_type="simple",
        lookback=pd.DateOffset(days=1),
        lag=pd.DateOffset(days=0),
        measure="mean",
    )

    assert algo(strategy)
    assert strategy.temp["selected"] == ["A"]
    assert strategy.temp["moving_average"]["A"] == pytest.approx(105.0)
    assert strategy.temp["moving_average"]["B"] == pytest.approx(95.0)
    assert algo.factor_stats["moving_average"] is algo.ma_algo.stats
    assert bool(algo.history.loc[prices.index[1], "A"]) is True
    assert bool(algo.history.loc[prices.index[1], "B"]) is False


def test_price_crossover_signal_respects_existing_candidate_pool():
    prices = _prices(A=[100.0, 110.0], B=[100.0, 90.0])
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["B"]
    algo = PriceCrossOverSignal(
        ma_type="simple",
        lookback=pd.DateOffset(days=1),
    )

    assert algo(strategy)
    assert strategy.temp["selected"] == []
    assert list(strategy.temp["moving_average"].index) == ["B"]
    assert bool(algo.history.loc[prices.index[1], "A"]) is False
    assert bool(algo.history.loc[prices.index[1], "B"]) is False


def test_price_crossover_signal_validates_ma_type():
    with pytest.raises(TypeError, match="ma_type"):
        PriceCrossOverSignal(ma_type=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ma_type"):
        PriceCrossOverSignal(ma_type="weighted")
    with pytest.raises(TypeError, match="half_life"):
        PriceCrossOverSignal(ma_type="exponential")


def test_price_crossover_signal_returns_false_when_internal_factor_cannot_compute():
    prices = _prices(A=[100.0, 101.0])
    strategy = _strategy_context(prices, now_idx=0)

    assert not PriceCrossOverSignal(
        ma_type="simple",
        lookback=pd.DateOffset(days=1),
        lag=pd.DateOffset(days=3),
    )(strategy)


def test_dual_ma_crossover_signal_selects_short_over_long():
    prices = _prices(
        A=[100.0, 100.0, 110.0], B=[100.0, 100.0, 90.0], C=[100.0, 100.0, 100.0]
    )
    strategy = _strategy_context(prices, now_idx=2)
    algo = DualMACrossoverSignal(
        short_ma_type="exponential",
        long_ma_type="simple",
        short_half_life=0.5,
        long_lookback=pd.DateOffset(days=1),
        long_measure="mean",
    )

    assert algo(strategy)
    assert strategy.temp["selected"] == ["A"]
    assert algo.factor_stats["short"] is algo.short_ma_algo.stats
    assert algo.factor_stats["long"] is algo.long_ma_algo.stats


def test_dual_ma_crossover_signal_validates_ma_types():
    with pytest.raises(TypeError, match="ma_type"):
        DualMACrossoverSignal(
            short_ma_type=None,  # type: ignore[arg-type]
            long_ma_type="simple",
        )
    with pytest.raises(ValueError, match="ma_type"):
        DualMACrossoverSignal(
            short_ma_type="weighted",
            long_ma_type="simple",
        )
    with pytest.raises(TypeError, match="half_life"):
        DualMACrossoverSignal(
            short_ma_type="exponential",
            long_ma_type="simple",
        )


def test_momentum_signal_validates_inputs():
    with pytest.raises(TypeError, match="ranking_algo"):
        MomentumSignal(ranking_algo=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="lookback"):
        MomentumSignal(
            ranking_algo=SelectN(n=1, stat_key="total_return"),
            lookback="1 day",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="lag"):
        MomentumSignal(
            ranking_algo=SelectN(n=1, stat_key="total_return"),
            lag=1,  # type: ignore[arg-type]
        )


def test_momentum_signal_uses_ranking_algo_to_choose_top_names():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 99.0], C=[100.0, 105.0])
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["A", "B", "C"]

    algo = MomentumSignal(
        ranking_algo=SelectN(n=1, stat_key="total_return", sort_descending=True),
        lookback=pd.DateOffset(days=1),
    )
    assert algo(strategy)
    assert strategy.temp["selected"] == ["C"]
    assert strategy.temp["total_return"]["A"] == pytest.approx(0.01)
    assert strategy.temp["total_return"]["B"] == pytest.approx(-0.01)
    assert strategy.temp["total_return"]["C"] == pytest.approx(0.05)
    assert algo.factor_stats["total_return"] is algo.total_return_algo.stats
    assert (
        int(algo.factor_stats["total_return"].loc[prices.index[1], "TOTAL_COVERED"])
        == 3
    )


def test_momentum_signal_respects_existing_candidate_pool_for_internal_factor():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 99.0], C=[100.0, 105.0])
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["B", "C"]

    algo = MomentumSignal(
        ranking_algo=SelectN(n=1, stat_key="total_return"),
        lookback=pd.DateOffset(days=1),
    )
    assert algo(strategy)
    assert strategy.temp["selected"] == ["C"]
    assert list(strategy.temp["total_return"].index) == ["B", "C"]


def test_momentum_signal_returns_false_when_internal_factor_cannot_compute():
    prices = _prices(A=[100.0, 101.0])
    strategy = _strategy_context(prices, now_idx=1)

    assert not MomentumSignal(
        ranking_algo=SelectN(n=1, stat_key="total_return"),
        lookback=pd.DateOffset(days=0),
        lag=pd.DateOffset(days=3),
    )(strategy)
