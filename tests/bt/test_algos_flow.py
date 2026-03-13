from types import SimpleNamespace

import pandas as pd
import pytest

from bt.algos.flow import (
    ClosePositionsAfterDates,
    Not,
    Or,
    Require,
    RunAfterDate,
    RunAfterDays,
    RunAfterMonths,
    RunDaily,
    RunEveryNMonths,
    RunEveryNPeriods,
    RunIfCashOutOfBounds,
    RunIfOutOfBounds,
    RunMonthly,
    RunOnDate,
    RunOnce,
    RunPeriod,
    RunQuarterly,
    RunWeekly,
    RunYearly,
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


def _period_target(
    index: pd.DatetimeIndex, now: pd.Timestamp | object
) -> SimpleNamespace:
    return SimpleNamespace(data=pd.DataFrame(index=index), now=now)


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


def test_require_not_and_or_cover_basic_boolean_composition():
    target = SimpleNamespace(temp={"selected": ["A"]})

    assert Require(bool, "selected")(target)
    assert Not(lambda _: False)(target)
    assert Or([lambda _: False, lambda _: True])(target)


def test_run_once_ignores_bootstrap_row_and_then_runs_only_once():
    algo = RunOnce()
    target = SimpleNamespace(inow=0)

    assert not algo(target)

    target.inow = 1
    assert algo(target)
    assert not algo(target)


def test_run_once_can_be_disabled_from_first_call():
    algo = RunOnce(run_on_first_call=False)

    assert [algo(SimpleNamespace(inow=1)) for _ in range(3)] == [False, False, False]


def test_run_period_uses_bootstrap_row_and_first_last_flags():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    default_algo = RunPeriod()

    assert not default_algo(_period_target(dates, dates[0]))
    assert default_algo(_period_target(dates, dates[1]))
    assert not default_algo(_period_target(dates, dates[-1]))

    end_algo = RunPeriod(
        run_on_first_date=False,
        run_on_end_of_period=True,
        run_on_last_date=True,
    )
    assert not end_algo(_period_target(dates, dates[1]))
    assert end_algo(_period_target(dates, dates[-1]))


def test_run_period_returns_false_on_missing_or_invalid_state():
    algo = RunDaily()

    assert not algo(SimpleNamespace())
    assert not algo(
        SimpleNamespace(
            data=pd.DataFrame(index=pd.date_range("2024-01-01", periods=2)), now="bad"
        )
    )


@pytest.mark.parametrize(
    ("algo_cls", "index", "triggered_now", "not_triggered_now"),
    [
        (RunDaily, pd.date_range("2024-01-01", periods=4, freq="D"), 2, 1),
        (RunWeekly, pd.date_range("2024-01-01", periods=12, freq="D"), 7, 6),
        (RunMonthly, pd.date_range("2024-01-01", periods=40, freq="D"), 31, 30),
        (RunQuarterly, pd.date_range("2024-01-01", periods=100, freq="D"), 91, 90),
        (RunYearly, pd.date_range("2024-01-01", periods=370, freq="D"), 366, 365),
    ],
)
def test_period_algos_detect_boundary_transitions(
    algo_cls,
    index: pd.DatetimeIndex,
    triggered_now: int,
    not_triggered_now: int,
):
    algo = algo_cls(run_on_first_date=False)

    assert not algo(_period_target(index, index[not_triggered_now]))
    assert algo(_period_target(index, index[triggered_now]))


def test_run_on_date_accepts_scalar_and_iterable_dates():
    scalar_algo = RunOnDate("2024-01-02")
    iterable_algo = RunOnDate(["2024-01-01", "2024-01-03"])

    assert scalar_algo(SimpleNamespace(now=pd.Timestamp("2024-01-02")))
    assert not scalar_algo(SimpleNamespace(now=pd.Timestamp("2024-01-03")))
    assert iterable_algo(SimpleNamespace(now=pd.Timestamp("2024-01-03")))


def test_run_on_date_validates_configuration_and_invalid_now():
    with pytest.raises(ValueError, match="valid timestamp"):
        RunOnDate([pd.NaT])

    algo = RunOnDate(["2024-01-01"])
    assert not algo(SimpleNamespace())
    assert not algo(SimpleNamespace(now="bad"))


def test_run_after_variants_gate_execution_as_expected():
    after_days = RunAfterDays(2)
    assert [after_days(None) for _ in range(4)] == [False, False, True, True]

    after_date = RunAfterDate("2024-01-03")
    assert not after_date(SimpleNamespace(now=pd.Timestamp("2024-01-03")))
    assert after_date(SimpleNamespace(now=pd.Timestamp("2024-01-04")))
    assert after_date(SimpleNamespace(now=pd.Timestamp("2024-01-04")))

    after_months = RunAfterMonths(2)
    assert not after_months(SimpleNamespace(now=pd.Timestamp("2024-01-15")))
    assert not after_months(SimpleNamespace(now=pd.Timestamp("2024-02-01")))
    assert after_months(SimpleNamespace(now=pd.Timestamp("2024-03-01")))


def test_run_every_n_periods_tracks_phase_and_deduplicates_same_day():
    algo = RunEveryNPeriods(n=3, offset=1)
    first = SimpleNamespace(now=pd.Timestamp("2024-01-01"))
    second = SimpleNamespace(now=pd.Timestamp("2024-01-02"))
    third = SimpleNamespace(now=pd.Timestamp("2024-01-03"))

    assert not algo(first)
    assert not algo(first)
    assert algo(second)
    assert not algo(third)


def test_run_every_n_periods_validates_inputs():
    with pytest.raises(ValueError, match="`n` must be > 0"):
        RunEveryNPeriods(0)
    with pytest.raises(ValueError, match="offset"):
        RunEveryNPeriods(2, offset=2)


def test_run_every_n_months_skips_bootstrap_and_uses_month_spacing():
    algo = RunEveryNMonths(2)

    assert not algo(SimpleNamespace(now=pd.Timestamp("2024-01-01"), inow=0))
    assert algo(SimpleNamespace(now=pd.Timestamp("2024-01-02"), inow=1))
    assert not algo(SimpleNamespace(now=pd.Timestamp("2024-01-02"), inow=1))
    assert not algo(SimpleNamespace(now=pd.Timestamp("2024-02-01"), inow=2))
    assert algo(SimpleNamespace(now=pd.Timestamp("2024-03-01"), inow=3))


def test_run_if_out_of_bounds_uses_cached_weights_after_first_valid_call():
    algo = RunIfOutOfBounds(0.5)
    target = SimpleNamespace(
        temp={"weights": {"c1": 0.5, "c2": 0.5}},
        children={
            "c1": SimpleNamespace(weight=0.5),
            "c2": SimpleNamespace(weight=0.5),
        },
    )

    assert not algo(target)

    target.temp = {}
    target.children["c1"].weight = 0.76
    target.children["c2"].weight = 0.24

    assert algo(target)


def test_run_if_out_of_bounds_supports_absolute_mode_and_cash_key():
    algo = RunIfOutOfBounds(0.1, mode="absolute")
    target = SimpleNamespace(
        temp={"weights": {"c1": 0.5, "cash": 0.0}},
        children={"c1": SimpleNamespace(weight=0.58)},
    )

    assert not algo(target)

    target.children["c1"].weight = 0.62
    assert algo(target)


def test_run_if_out_of_bounds_validates_inputs_and_missing_state():
    with pytest.raises(ValueError, match=">= 0"):
        RunIfOutOfBounds(-0.1)
    with pytest.raises(ValueError, match="mode"):
        RunIfOutOfBounds(0.1, mode="bad")

    algo = RunIfOutOfBounds(0.1)
    assert not algo(SimpleNamespace())
    assert not algo(SimpleNamespace(temp={}))
    assert not algo(SimpleNamespace(temp={"weights": object()}))


def test_run_if_cash_out_of_bounds_checks_cash_weight():
    algo = RunIfCashOutOfBounds(0.05)
    target = SimpleNamespace(
        temp={"cash": 0.10},
        capital=100.0,
        value=1_000.0,
    )

    assert not algo(target)

    target.capital = 200.0
    assert algo(target)


def test_run_if_cash_out_of_bounds_validates_and_handles_missing_state():
    with pytest.raises(ValueError, match=">= 0"):
        RunIfCashOutOfBounds(-0.01)

    algo = RunIfCashOutOfBounds(0.01)
    assert not algo(SimpleNamespace())
    assert not algo(SimpleNamespace(temp={}, capital=100.0, value=1_000.0))


def test_close_positions_after_dates_closes_matching_security_children():
    prices = _prices(A=[100.0, 100.0, 100.0], B=[100.0, 100.0, 100.0])
    strategy = Strategy("s", children=[Security("A"), Security("B")])
    _prepare_strategy_for_trading(strategy, prices)
    strategy.allocate(500.0, child="A")
    strategy.allocate(500.0, child="B")
    strategy.post_market_update()
    strategy.pre_market_update(prices.index[2], 2)

    algo = ClosePositionsAfterDates(
        pd.Series({"A": prices.index[2], "B": prices.index[2] + pd.Timedelta(days=1)})
    )

    assert algo(strategy)
    assert strategy.children["A"].position == 0.0
    assert strategy.children["B"].position == 5.0
    assert strategy.perm["closed"] == {"A"}


def test_close_positions_after_dates_resolves_named_setup_data():
    prices = _prices(A=[100.0, 100.0, 100.0])
    strategy = Strategy("s", children=[Security("A")])
    strategy.setup(prices, last_valid_date=pd.Series({"A": prices.index[2]}))
    strategy.adjust(1_000.0)
    strategy.pre_market_update(prices.index[0], 0)
    strategy.post_market_update()
    strategy.pre_market_update(prices.index[1], 1)
    strategy.allocate(1_000.0, child="A")
    strategy.post_market_update()
    strategy.pre_market_update(prices.index[2], 2)

    algo = ClosePositionsAfterDates("last_valid_date")

    assert algo(strategy)
    assert strategy.children["A"].position == 0.0
