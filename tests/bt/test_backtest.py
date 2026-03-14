from unittest import mock

import numpy as np
import pandas as pd
import pytest

from bt.algos.core import Algo
from bt.algos.flow import RunOnce
from bt.algos.portfolio_ops import Rebalance
from bt.algos.weighting import WeightFixed
from bt.core import Backtest, LongShortStrategy, Strategy


class RecordRunState(Algo):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[pd.Timestamp, int]] = []

    def __call__(self, target) -> bool:
        self.calls.append((pd.Timestamp(target.now), int(target.inow)))
        return True


class ReadAdditionalData(Algo):
    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        self.shape: tuple[int, int] | None = None

    def __call__(self, target) -> bool:
        self.shape = target.get_data(self.key).shape
        return True


class SetCashReserve(Algo):
    def __init__(self, cash_fraction: float) -> None:
        super().__init__()
        self.cash_fraction = float(cash_fraction)

    def __call__(self, target) -> bool:
        target.temp["cash"] = self.cash_fraction
        return True


def _prices(**columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        columns,
        index=pd.date_range(
            "2024-01-01", periods=len(next(iter(columns.values()))), freq="D"
        ),
        dtype=float,
    )


def test_backtest_copies_strategy():
    strategy = Strategy("copy_test")
    prices = _prices(A=[100.0, 101.0])

    backtest = Backtest(strategy, prices, progress_bar=False)

    assert backtest.strategy is not strategy
    assert backtest.strategy.name == strategy.name


def test_backtest_uses_strategy_name_by_default():
    prices = _prices(A=[100.0, 101.0])

    backtest = Backtest(Strategy("named"), prices, progress_bar=False)

    assert backtest.name == "named"


def test_backtest_defaults_to_fractional_positions():
    prices = _prices(A=[100.0, 101.0])

    backtest = Backtest(Strategy("fractional_default"), prices, progress_bar=False)

    assert backtest.strategy.integer_positions is False


def test_backtest_validates_primary_data_inputs():
    strategy = Strategy("test")

    with pytest.raises(TypeError, match="pandas DataFrame"):
        Backtest(strategy, [1, 2, 3])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty"):
        Backtest(strategy, pd.DataFrame())

    with pytest.raises(TypeError, match="DatetimeIndex"):
        Backtest(strategy, pd.DataFrame({"A": [1.0, 2.0]}, index=[0, 1]))

    duplicate_index = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    with pytest.raises(ValueError, match="unique"):
        Backtest(strategy, pd.DataFrame({"A": [1.0, 2.0]}, index=duplicate_index))

    descending_index = pd.DatetimeIndex(["2024-01-02", "2024-01-01"])
    with pytest.raises(ValueError, match="sorted"):
        Backtest(strategy, pd.DataFrame({"A": [1.0, 2.0]}, index=descending_index))

    duplicate_columns = pd.DataFrame(
        np.ones((2, 2)),
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
        columns=["A", "A"],
    )
    with pytest.raises(ValueError, match="duplicate columns"):
        Backtest(strategy, duplicate_columns)


def test_backtest_validates_initial_capital_is_finite():
    prices = _prices(A=[100.0, 101.0])

    with pytest.raises(ValueError, match="finite"):
        Backtest(Strategy("test"), prices, initial_capital=np.inf)


def test_backtest_validates_live_start_date():
    prices = _prices(A=[100.0, 101.0, 102.0])

    with pytest.raises(ValueError, match="parseable as a timestamp"):
        Backtest(Strategy("test"), prices, live_start_date="bad-date")

    with pytest.raises(ValueError, match="present in the prices index"):
        Backtest(Strategy("test"), prices, live_start_date="2025-01-01")


def test_backtest_runs_only_from_second_date():
    recorder = RecordRunState()
    prices = _prices(A=[100.0, 101.0, 102.0, 103.0])
    backtest = Backtest(
        Strategy("run_state", algos=[recorder]), prices, progress_bar=False
    )

    backtest.run()

    copied_recorder = backtest.strategy.algos["RecordRunState"]
    assert copied_recorder.calls == [
        (prices.index[1], 1),
        (prices.index[2], 2),
        (prices.index[3], 3),
    ]


def test_backtest_seeds_to_live_start_date_and_runs_after_it():
    recorder = RecordRunState()
    prices = _prices(A=[100.0, 101.0, 102.0, 103.0])
    backtest = Backtest(
        Strategy("run_state", algos=[recorder]),
        prices,
        live_start_date=prices.index[2],
        progress_bar=False,
    )

    backtest.run()

    copied_recorder = backtest.strategy.algos["RecordRunState"]
    assert copied_recorder.calls == [(prices.index[3], 3)]
    assert backtest.strategy.values.index[0] == prices.index[2]
    assert list(backtest.strategy.data.index) == list(prices.index[2:])


def test_backtest_trades_pre_market_using_prior_close():
    prices = _prices(A=[10.0, 20.0, 20.0])
    strategy = Strategy("prior_close", algos=[WeightFixed(A=1.0), Rebalance()])
    backtest = Backtest(strategy, prices, initial_capital=1_000.0, progress_bar=False)

    backtest.run()

    traded_day = prices.index[1]
    closed_day = prices.index[2]
    assert backtest.positions.loc[traded_day, "A"] == 100.0
    assert backtest.positions.loc[closed_day, "A"] == 100.0
    assert backtest.strategy.children["A"].position == 100.0


def test_backtest_live_start_date_uses_live_window_prior_close():
    prices = _prices(A=[10.0, 20.0, 30.0, 30.0])
    strategy = Strategy("prior_close", algos=[WeightFixed(A=1.0), Rebalance()])
    backtest = Backtest(
        strategy,
        prices,
        initial_capital=1_000.0,
        live_start_date=prices.index[2],
        integer_positions=False,
        progress_bar=False,
    )

    backtest.run()

    traded_day = prices.index[3]
    assert backtest.positions.loc[traded_day, "A"] == pytest.approx(1000.0 / 30.0)


def test_backtest_creates_missing_security_children_on_demand():
    prices = _prices(A=[100.0, 101.0, 102.0])
    strategy = Strategy("dynamic_children", algos=[WeightFixed(A=1.0), Rebalance()])
    backtest = Backtest(strategy, prices, progress_bar=False)

    backtest.run()

    assert "A" in backtest.strategy.children
    assert backtest.strategy.children["A"]._issec is True
    assert backtest.positions["A"].max() > 0
    assert backtest.positions["A"].iloc[-1] > 0.0


def test_backtest_forwards_additional_data():
    reader = ReadAdditionalData("signal")
    prices = _prices(A=[100.0, 101.0, 102.0])
    signal = pd.DataFrame(
        {"flag": [True, False, True]},
        index=prices.index,
    )
    backtest = Backtest(
        Strategy("with_data", algos=[reader]),
        prices,
        additional_data={"signal": signal},
        progress_bar=False,
    )

    backtest.run()

    copied_reader = backtest.strategy.algos["ReadAdditionalData"]
    assert copied_reader.shape == signal.shape


def test_backtest_transactions_property_returns_multiindex_frame():
    prices = _prices(A=[100.0, 101.0, 102.0])
    strategy = Strategy("transactions", algos=[WeightFixed(A=1.0), Rebalance()])
    backtest = Backtest(strategy, prices, progress_bar=False)

    backtest.run()

    transactions = backtest.get_transactions
    assert isinstance(transactions, pd.DataFrame)
    assert list(transactions.columns) == ["price", "quantity"]
    assert transactions.index.names == ["Date", "Security"]
    assert len(transactions) == 1
    assert transactions["quantity"].iloc[0] > 0


def test_backtest_exposes_weights_security_weights_and_turnover():
    prices = _prices(A=[100.0, 101.0, 102.0], B=[100.0, 99.0, 98.0])
    strategy = Strategy(
        "analytics",
        algos=[WeightFixed(A=0.5, B=0.5), Rebalance()],
    )
    backtest = Backtest(strategy, prices, integer_positions=False, progress_bar=False)

    backtest.run()

    assert list(backtest.weights.columns) == ["analytics", "analytics>A", "analytics>B"]
    assert list(backtest.security_weights.columns) == ["Cash", "A", "B"]
    assert isinstance(backtest.turnover, pd.Series)
    assert backtest.turnover.index.equals(backtest.strategy.values.index)


def test_root_bankruptcy_stops_after_bankrupt_close():
    prices = _prices(A=[100.0, -200.0, 50.0])
    strategy = Strategy("root_bankrupt", algos=[WeightFixed(A=1.0), Rebalance()])
    backtest = Backtest(strategy, prices, progress_bar=False)

    backtest.run()

    assert backtest.strategy.bankrupt is True
    assert backtest.strategy.perm["closed"] == {"root_bankrupt"}
    assert backtest.strategy.now == prices.index[1]
    assert backtest.strategy.values.index[-1] == prices.index[1]
    assert prices.index[2] not in backtest.strategy.values.index


def test_child_strategy_bankruptcy_becomes_nan_and_parent_continues():
    prices = _prices(A=[100.0, -200.0, 50.0])
    child = Strategy("child", algos=[WeightFixed(A=1.0), Rebalance()])
    parent = Strategy(
        "parent",
        algos=[SetCashReserve(0.8), WeightFixed(child=1.0), Rebalance()],
        children=[child],
    )
    backtest = Backtest(
        parent,
        prices,
        initial_capital=100.0,
        integer_positions=False,
        progress_bar=False,
    )

    backtest.run()

    copied_child = backtest.strategy.children["child"]
    bankrupt_day = prices.index[1]
    next_day = prices.index[2]

    assert copied_child.bankrupt is True
    assert copied_child._bankrupt_at == bankrupt_day
    assert copied_child.perm["closed"] == {"child"}
    assert pd.notna(copied_child.data.loc[bankrupt_day, "value"])
    assert pd.isna(copied_child.data.loc[next_day, "price"])
    assert pd.isna(copied_child.data.loc[next_day, "value"])
    assert copied_child.value == 0.0
    assert backtest.strategy.bankrupt is False
    assert backtest.strategy.values.index[-1] == next_day
    assert pd.notna(backtest.strategy.data.loc[next_day, "value"])


def test_backtest_failed_run_does_not_mark_has_run():
    strategy = mock.MagicMock()
    strategy.name = "test"
    strategy.use_integer_positions.return_value = None
    strategy.set_commissions.return_value = None
    strategy.setup.side_effect = RuntimeError("boom")
    prices = _prices(A=[100.0, 101.0])

    backtest = Backtest(strategy, prices, progress_bar=False)

    with pytest.raises(RuntimeError, match="boom"):
        backtest.run()

    assert backtest.has_run is False


def test_long_short_strategy_executes_one_signed_top_level_book():
    prices = _prices(A=[100.0, 100.0, 100.0], B=[100.0, 100.0, 100.0])
    long_strategy = Strategy("long_model", algos=[WeightFixed(A=1.0), Rebalance()])
    short_strategy = Strategy("short_model", algos=[WeightFixed(B=1.0), Rebalance()])
    strategy = LongShortStrategy(
        "ls",
        long_strategy=long_strategy,
        short_strategy=short_strategy,
        long_exposure=1.0,
        short_exposure=1.0,
    )
    backtest = Backtest(
        strategy,
        prices,
        initial_capital=1_000.0,
        integer_positions=False,
        progress_bar=False,
    )

    backtest.run()

    traded_day = prices.index[1]
    assert backtest.positions.loc[traded_day, "A"] == pytest.approx(10.0)
    assert backtest.positions.loc[traded_day, "B"] == pytest.approx(-10.0)
    assert backtest.strategy.children["A"].parent is backtest.strategy
    assert backtest.strategy.children["B"].parent is backtest.strategy
    assert "long_model" not in backtest.strategy.children
    assert "short_model" not in backtest.strategy.children


def test_long_short_strategy_nets_overlapping_long_and_short_targets():
    prices = _prices(A=[100.0, 100.0, 100.0])
    strategy = LongShortStrategy(
        "ls",
        long_strategy=Strategy("long_model", algos=[WeightFixed(A=1.0)]),
        short_strategy=Strategy("short_model", algos=[WeightFixed(A=1.0)]),
    )
    backtest = Backtest(
        strategy,
        prices,
        initial_capital=1_000.0,
        integer_positions=False,
        progress_bar=False,
    )

    backtest.run()

    assert "A" not in backtest.strategy.children
    assert backtest.get_transactions.empty


def test_long_short_strategy_caches_latest_sleeve_weights_between_emissions():
    prices = _prices(A=[100.0, 100.0, 100.0, 100.0], B=[100.0, 100.0, 100.0, 100.0])
    strategy = LongShortStrategy(
        "ls",
        long_strategy=Strategy("long_model", algos=[RunOnce(), WeightFixed(A=1.0)]),
        short_strategy=Strategy("short_model", algos=[RunOnce(), WeightFixed(B=1.0)]),
    )
    backtest = Backtest(
        strategy,
        prices,
        initial_capital=1_000.0,
        integer_positions=False,
        progress_bar=False,
    )

    backtest.run()

    first_run_day = prices.index[1]
    second_run_day = prices.index[2]

    assert backtest.positions.loc[first_run_day, "A"] == pytest.approx(10.0)
    assert backtest.positions.loc[first_run_day, "B"] == pytest.approx(-10.0)
    assert backtest.positions.loc[second_run_day, "A"] == pytest.approx(10.0)
    assert backtest.positions.loc[second_run_day, "B"] == pytest.approx(-10.0)
    assert backtest.strategy._long_weights == {"A": 1.0}
    assert backtest.strategy._short_weights == {"B": 1.0}
