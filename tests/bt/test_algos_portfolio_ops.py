from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from bt.algos.portfolio_ops import Rebalance, RebalanceOverTime
from bt.core import Strategy


def _prices(**columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        columns,
        index=pd.date_range(
            "2024-01-01", periods=len(next(iter(columns.values()))), freq="D"
        ),
        dtype=float,
    )


def _prepare_strategy_for_trading(
    strategy: Strategy,
    prices: pd.DataFrame,
    capital: float = 1_000.0,
) -> None:
    strategy.setup(prices)
    strategy.adjust(capital)
    strategy.pre_market_update(prices.index[0], 0)
    strategy.post_market_update()
    strategy.pre_market_update(prices.index[1], 1)


def test_rebalance_allocates_to_target_weights_and_creates_children():
    algo = Rebalance()
    prices = _prices(c1=[100.0, 100.0], c2=[100.0, 100.0])
    strategy = Strategy("s")
    _prepare_strategy_for_trading(strategy, prices)

    strategy.temp["weights"] = {"c1": 1.0}

    assert algo(strategy)

    strategy.post_market_update()

    assert strategy.capital == 0.0
    assert strategy.value == 1_000.0
    assert strategy["c1"].position == 10.0
    assert strategy["c1"].value == 1_000.0
    assert strategy["c1"].weight == 1.0


def test_rebalance_with_commissions_updates_close_state():
    algo = Rebalance()
    prices = _prices(c1=[100.0, 100.0], c2=[100.0, 100.0])
    strategy = Strategy("s")
    strategy.set_commissions(lambda q, p: 1.0)
    _prepare_strategy_for_trading(strategy, prices)

    strategy.temp["weights"] = {"c1": 1.0}

    assert algo(strategy)

    strategy.post_market_update()

    assert strategy.capital == 99.0
    assert strategy.value == 999.0
    assert strategy.fees.loc[prices.index[1]] == 1.0
    assert strategy["c1"].position == 9.0
    assert strategy["c1"].value == 900.0
    assert strategy["c1"].weight == pytest.approx(900.0 / 999.0)


def test_rebalance_with_cash_reserve_leaves_cash_unallocated():
    algo = Rebalance()
    prices = _prices(c1=[100.0, 100.0], c2=[100.0, 100.0])
    strategy = Strategy("s")
    _prepare_strategy_for_trading(strategy, prices)

    strategy.temp["weights"] = {"c1": 1.0}
    strategy.temp["cash"] = 0.5

    assert algo(strategy)

    strategy.post_market_update()

    assert strategy.capital == 500.0
    assert strategy.value == 1_000.0
    assert strategy["c1"].position == 5.0
    assert strategy["c1"].weight == 0.5


def test_rebalance_closes_children_missing_from_new_targets():
    algo = Rebalance()
    prices = _prices(c1=[100.0, 100.0, 100.0], c2=[100.0, 100.0, 100.0])
    strategy = Strategy("s")
    _prepare_strategy_for_trading(strategy, prices)

    strategy.temp["weights"] = {"c1": 1.0}
    assert algo(strategy)
    strategy.post_market_update()
    strategy.pre_market_update(prices.index[2], 2)

    strategy.temp["weights"] = {"c2": 1.0}
    assert algo(strategy)

    strategy.post_market_update()

    assert strategy["c1"].position == 0.0
    assert strategy["c1"].value == 0.0
    assert strategy["c2"].position == 10.0
    assert strategy.positions.loc[prices.index[2], "c1"] == 0.0
    assert strategy.positions.loc[prices.index[2], "c2"] == 10.0


def test_rebalance_accepts_series_weights_and_drops_nan_targets():
    algo = Rebalance()
    prices = _prices(c1=[100.0, 100.0], c2=[100.0, 100.0])
    strategy = Strategy("s")
    _prepare_strategy_for_trading(strategy, prices)

    strategy.temp["weights"] = pd.Series({"c1": 1.0, "c2": None})

    assert algo(strategy)

    strategy.post_market_update()

    assert "c1" in strategy.children
    assert "c2" not in strategy.children
    assert strategy["c1"].position == 10.0


def test_rebalance_returns_false_for_invalid_weight_shapes():
    algo = Rebalance()
    prices = _prices(c1=[100.0, 100.0])
    strategy = Strategy("s")
    _prepare_strategy_for_trading(strategy, prices)

    strategy.temp["weights"] = ["c1"]

    assert not algo(strategy)


@pytest.mark.parametrize("cash", [-0.1, 1.1, "bad"])
def test_rebalance_rejects_invalid_cash_fraction(cash):
    algo = Rebalance()
    prices = _prices(c1=[100.0, 100.0])
    strategy = Strategy("s")
    _prepare_strategy_for_trading(strategy, prices)

    strategy.temp["weights"] = {"c1": 1.0}
    strategy.temp["cash"] = cash

    assert not algo(strategy)


def test_rebalance_over_time_spreads_target_weights_across_calls():
    algo = RebalanceOverTime(n=2)
    algo._rb = mock.MagicMock(return_value=True)

    target = SimpleNamespace(
        temp={"weights": {"a": 1.0, "b": 0.0}},
        children={
            "a": SimpleNamespace(_weight=0.0),
            "b": SimpleNamespace(_weight=1.0),
        },
    )

    assert algo(target)
    assert target.temp["weights"] == {"a": 0.5, "b": 0.5}

    target.children["a"]._weight = 0.5
    target.children["b"]._weight = 0.5
    target.temp = {}

    assert algo(target)
    assert target.temp["weights"] == {"a": 1.0, "b": 0.0}
    assert algo._rb.call_count == 2

    target.children["a"]._weight = 1.0
    target.children["b"]._weight = 0.0
    target.temp = {}

    assert algo(target)
    assert algo._rb.call_count == 2


def test_rebalance_over_time_validates_configuration_and_weight_shapes():
    with pytest.raises(TypeError, match="`n`"):
        RebalanceOverTime(n=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`n` must be > 0"):
        RebalanceOverTime(n=0)

    algo = RebalanceOverTime(n=2)
    target = SimpleNamespace(temp={"weights": ["a", "b"]}, children={})
    assert not algo(target)
