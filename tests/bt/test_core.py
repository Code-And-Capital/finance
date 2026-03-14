import pandas as pd
import pytest

from bt.algos.flow import ClosePositionsAfterDates, RunDaily, RunOnce
from bt.algos.portfolio_ops import Rebalance
from bt.algos.weighting import WeightFixed
from bt.core import (
    LongShortStrategy,
    Node,
    Security,
    Strategy,
    fixed_per_trade_commission,
    notional_bps_commission,
)


def _prices(**columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        columns,
        index=pd.date_range(
            "2024-01-01", periods=len(next(iter(columns.values()))), freq="D"
        ),
        dtype=float,
    )


def _seed_first_day(
    strategy: Strategy, prices: pd.DataFrame, capital: float = 1_000.0
) -> None:
    strategy.setup(prices)
    strategy.adjust(capital)
    first_day = prices.index[0]
    strategy.pre_market_update(first_day, 0)
    strategy.post_market_update()


def test_node_attaches_children_and_sets_parent_relationships():
    child_a = Node("a")
    child_b = Node("b")
    parent = Node("parent", children=[child_a, child_b])

    assert list(parent.children) == ["a", "b"]
    assert parent.children["a"].parent is parent
    assert parent.children["b"].parent is parent
    assert parent.children["a"].full_name == "parent>a"


def test_node_mapping_input_can_rename_child():
    original = Node("original")
    parent = Node("parent", children={"renamed": original})

    assert "renamed" in parent.children
    assert parent.children["renamed"].name == "renamed"
    assert parent.children["renamed"].parent is parent


def test_node_rejects_duplicate_child_names():
    with pytest.raises(ValueError, match="already exists"):
        Node("parent", children=[Node("dup"), Node("dup")])


def test_node_rejects_reattaching_child_to_second_parent():
    child = Node("child")
    first = Node("first", children=[child])

    with pytest.raises(ValueError, match="already attached"):
        Node("second", children=[first.children["child"]])


def test_use_integer_positions_propagates_to_attached_children():
    child = Node("child")
    parent = Node("parent", children=[child])

    parent.use_integer_positions(False)

    assert parent.integer_positions is False
    assert parent.children["child"].integer_positions is False


def test_security_setup_requires_prices_for_referenced_name():
    security = Security("A")
    prices = _prices(B=[100.0, 101.0])

    with pytest.raises(ValueError, match="Missing prices"):
        security.setup(prices)


def test_security_allocate_rounds_negative_integer_trades_toward_zero():
    prices = _prices(A=[100.0, 100.0])
    strategy = Strategy("parent", children=[Security("A")])
    _seed_first_day(strategy, prices, capital=0.0)

    security = strategy.children["A"]
    security._position = 10.0
    security._value = 1_000.0
    strategy.pre_market_update(prices.index[1], 1)

    security.allocate(-550.0)

    assert security.position == 5.0


def test_security_allocate_keeps_integer_buys_feasible_with_commissions():
    prices = _prices(A=[100.0, 100.0])
    strategy = Strategy("parent", children=[Security("A")])
    strategy.set_commissions(
        lambda quantity, price: fixed_per_trade_commission(
            quantity,
            price,
            fixed_fee=10.0,
        )
    )
    _seed_first_day(strategy, prices, capital=1_000.0)

    security = strategy.children["A"]
    strategy.pre_market_update(prices.index[1], 1)

    security.allocate(1_000.0)

    assert security.position == 9.0
    assert strategy.capital == pytest.approx(90.0)


def test_security_allocate_with_notional_bps_commission_does_not_raise_or_overspend():
    prices = _prices(A=[100.0, 100.0])
    strategy = Strategy("parent", children=[Security("A")])
    strategy.set_commissions(notional_bps_commission)
    _seed_first_day(strategy, prices, capital=1_000.0)

    security = strategy.children["A"]
    strategy.pre_market_update(prices.index[1], 1)

    security.allocate(1_000.0)

    assert security.position == 9.0
    assert strategy.capital == pytest.approx(97.75)


def test_strategy_ensure_child_creates_security_after_setup_and_seeds_prior_close():
    prices = _prices(A=[100.0, 120.0, 130.0])
    strategy = Strategy("dynamic")
    _seed_first_day(strategy, prices)
    strategy.pre_market_update(prices.index[1], 1)

    child = strategy._ensure_child("A")

    assert "A" in strategy.children
    assert child.parent is strategy
    assert child.now == prices.index[1]
    assert child.price == 100.0


def test_strategy_positions_aggregate_duplicate_security_names_across_child_strategies():
    prices = _prices(A=[100.0, 100.0])
    child_1 = Strategy("child_1", children=[Security("A")])
    child_2 = Strategy("child_2", children=[Security("A")])
    parent = Strategy("parent", children=[child_1, child_2])

    parent.setup(prices)
    parent.adjust(200.0)
    parent.pre_market_update(prices.index[0], 0)
    parent.post_market_update()
    parent.pre_market_update(prices.index[1], 1)

    parent.allocate(100.0, child="child_1")
    parent.allocate(100.0, child="child_2")
    parent.children["child_1"].children["A"].allocate(100.0)
    parent.children["child_2"].children["A"].allocate(100.0)
    parent.post_market_update()

    assert list(parent.positions.columns) == ["A"]
    assert parent.positions.loc[prices.index[1], "A"] == 2.0
    assert parent.outlays.loc[prices.index[1], "A"] == 200.0


def test_rebalance_executes_sells_before_buys_and_uses_target_dollars():
    prices = _prices(A=[100.0, 100.0, 100.0], B=[100.0, 100.0, 100.0])
    strategy = Strategy("parent", children=[Security("A"), Security("B")])
    _seed_first_day(strategy, prices, capital=1_000.0)

    strategy.pre_market_update(prices.index[1], 1)
    strategy.children["A"].allocate(500.0)
    strategy.children["B"].allocate(500.0)
    strategy.post_market_update()

    strategy.temp["weights"] = {"B": 0.6, "A": 0.4}
    strategy.pre_market_update(prices.index[2], 2)

    Rebalance()(strategy)

    assert strategy.capital == pytest.approx(0.0)
    assert strategy.children["A"].position == pytest.approx(4.0)
    assert strategy.children["B"].position == pytest.approx(6.0)


def test_strategy_securities_returns_all_descendant_security_nodes():
    prices = _prices(A=[100.0, 100.0], B=[50.0, 50.0])
    child = Strategy("child", children=[Security("A")])
    parent = Strategy("parent", children=[child, Security("B")])
    parent.setup(prices)

    names = sorted(security.name for security in parent.securities)

    assert names == ["A", "B"]


def test_strategy_prepends_close_positions_after_dates_by_default():
    strategy = Strategy("s", algos=[RunOnce()])

    assert isinstance(strategy.stack.algos[0], ClosePositionsAfterDates)
    assert strategy.stack.algos[0]._close_name == "last_valid_date"
    assert isinstance(strategy.stack.algos[1], RunOnce)


def test_strategy_dedupes_manual_last_valid_date_close_algo():
    close_algo = ClosePositionsAfterDates("last_valid_date")
    strategy = Strategy("s", algos=[close_algo, RunOnce()])

    close_algos = [
        algo
        for algo in strategy.stack.algos
        if isinstance(algo, ClosePositionsAfterDates)
        and algo._close_name == "last_valid_date"
    ]

    assert len(close_algos) == 1


def test_long_short_strategy_strips_execution_algos_from_model_sleeves():
    long_strategy = Strategy("long", algos=[WeightFixed(A=1.0), Rebalance()])
    short_strategy = Strategy("short", algos=[WeightFixed(B=1.0), Rebalance()])

    strategy = LongShortStrategy(
        "ls",
        long_strategy=long_strategy,
        short_strategy=short_strategy,
    )

    assert all(
        not isinstance(algo, Rebalance) for algo in strategy.long_strategy.stack.algos
    )
    assert all(
        not isinstance(algo, Rebalance) for algo in strategy.short_strategy.stack.algos
    )
    assert "Rebalance" not in strategy.long_strategy.algos
    assert "Rebalance" not in strategy.short_strategy.algos
    assert isinstance(strategy.stack.algos[1], RunDaily)
    assert isinstance(strategy.stack.algos[2], Rebalance)
    assert strategy.long_strategy is not long_strategy
    assert strategy.short_strategy is not short_strategy
    assert any(isinstance(algo, Rebalance) for algo in long_strategy.stack.algos)
    assert any(isinstance(algo, Rebalance) for algo in short_strategy.stack.algos)


def test_long_short_strategy_requires_distinct_sleeve_instances():
    sleeve = Strategy("sleeve", algos=[WeightFixed(A=1.0)])

    with pytest.raises(ValueError, match="must be different"):
        LongShortStrategy(
            "ls",
            long_strategy=sleeve,
            short_strategy=sleeve,
        )


def test_long_short_strategy_accepts_custom_frequency_algo():
    strategy = LongShortStrategy(
        "ls",
        long_strategy=Strategy("long", algos=[WeightFixed(A=1.0)]),
        short_strategy=Strategy("short", algos=[WeightFixed(B=1.0)]),
        frequency_algo=RunOnce(),
    )

    assert isinstance(strategy.stack.algos[1], RunOnce)
    assert isinstance(strategy.stack.algos[2], Rebalance)
