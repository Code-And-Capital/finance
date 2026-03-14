from types import SimpleNamespace

import pandas as pd
import pytest

from bt.algos.factors import (
    ExponentialWeightedMovingAverage,
    KernelMovingAverage,
    SetFactor,
    SimpleMovingAverage,
    TotalReturn,
)
from bt.analytics import BacktestSummary
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


def test_set_factor_reads_market_data_at_last_day():
    prices = _prices(A=[100.0, 101.0, 102.0])
    factor_data = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0]},
        index=prices.index,
    )
    strategy = _strategy_context(
        prices,
        now_idx=2,
        last_day_idx=1,
        quality=factor_data,
    )

    assert SetFactor("quality")(strategy)
    assert strategy.temp["quality"]["A"] == pytest.approx(2.0)


def test_set_factor_can_standardize_cross_section():
    prices = _prices(A=[100.0, 101.0], B=[100.0, 101.0], C=[100.0, 101.0])
    factor_data = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0]},
        index=prices.index,
    )
    strategy = _strategy_context(prices, now_idx=1, quality=factor_data)

    assert SetFactor("quality", standardize=True)(strategy)
    assert strategy.temp["quality"]["A"] == pytest.approx(-1.2247448714)
    assert strategy.temp["quality"]["B"] == pytest.approx(0.0)
    assert strategy.temp["quality"]["C"] == pytest.approx(1.2247448714)


def test_set_factor_validates_standardize_flag():
    with pytest.raises(TypeError, match="standardize"):
        SetFactor("quality", standardize="yes")  # type: ignore[arg-type]


def test_simple_moving_average_computes_mean_and_median():
    prices = _prices(A=[1.0, 2.0, 3.0], B=[2.0, 4.0, 8.0])
    strategy = _strategy_context(prices, now_idx=2)

    mean_algo = SimpleMovingAverage(lookback=pd.DateOffset(days=2), measure="mean")
    median_algo = SimpleMovingAverage(lookback=pd.DateOffset(days=2), measure="median")

    assert mean_algo(strategy)
    assert strategy.temp["moving_average"]["A"] == pytest.approx(2.0)
    assert strategy.temp["moving_average"]["B"] == pytest.approx(14.0 / 3.0)

    assert median_algo(strategy)
    assert strategy.temp["moving_average"]["A"] == pytest.approx(2.0)
    assert strategy.temp["moving_average"]["B"] == pytest.approx(4.0)


def test_simple_moving_average_respects_existing_selected_pool():
    prices = _prices(A=[1.0, 2.0], B=[2.0, 4.0])
    strategy = _strategy_context(prices, now_idx=1)
    strategy.temp["selected"] = ["B"]

    assert SimpleMovingAverage(lookback=pd.DateOffset(days=1))(strategy)
    assert list(strategy.temp["moving_average"].index) == ["B"]


def test_simple_moving_average_does_not_accept_standardize_flag():
    with pytest.raises(TypeError, match="standardize"):
        SimpleMovingAverage(
            lookback=pd.DateOffset(days=1),
            standardize=True,  # type: ignore[call-arg]
        )


def test_exponential_weighted_moving_average_validates_and_computes():
    with pytest.raises(TypeError, match="half_life"):
        ExponentialWeightedMovingAverage(half_life="1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be > 0"):
        ExponentialWeightedMovingAverage(half_life=0)

    prices = _prices(A=[1.0, 3.0, 5.0])
    strategy = _strategy_context(prices, now_idx=2)

    algo = ExponentialWeightedMovingAverage(half_life=1.0)
    assert algo(strategy)
    assert strategy.temp["ewma"]["A"] == pytest.approx(3.5)


def test_kernel_moving_average_matches_linear_kernel():
    prices = _prices(A=[1.0, 2.0, 3.0, 4.0], B=[2.0, 2.0, 2.0, 2.0])
    strategy = _strategy_context(prices, now_idx=3)
    strategy.temp["selected"] = ["A", "B"]

    algo = KernelMovingAverage(lookback=pd.DateOffset(days=2), kernel_factor=1)
    assert algo(strategy)
    assert strategy.temp["kernel_moving_average"]["A"] == pytest.approx(
        (2 + 2 * 3 + 3 * 4) / 6
    )
    assert strategy.temp["kernel_moving_average"]["B"] == pytest.approx(2.0)


def test_kernel_moving_average_validates_inputs():
    with pytest.raises(TypeError, match="lookback"):
        KernelMovingAverage(lookback=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kernel_factor"):
        KernelMovingAverage(lookback=pd.DateOffset(days=3), kernel_factor=-1)


def test_total_return_uses_last_day_and_updates_stats():
    prices = _prices(A=[100.0, 110.0, 120.0], B=[100.0, 95.0, 90.0])
    strategy = _strategy_context(prices, now_idx=2, last_day_idx=1)
    strategy.temp["selected"] = ["A", "B"]

    algo = TotalReturn(lookback=pd.DateOffset(days=1))
    assert algo(strategy)

    total_return = strategy.temp["total_return"]
    assert total_return["A"] == pytest.approx(0.1)
    assert total_return["B"] == pytest.approx(-0.05)
    stats_row = algo.stats.loc[prices.index[1]]
    assert int(stats_row["TOTAL_COVERED"]) == 2


def test_total_return_can_store_standardized_cross_section():
    prices = _prices(A=[100.0, 110.0, 120.0], B=[100.0, 95.0, 90.0])
    strategy = _strategy_context(prices, now_idx=2, last_day_idx=1)
    strategy.temp["selected"] = ["A", "B"]

    algo = TotalReturn(lookback=pd.DateOffset(days=1), standardize=True)
    assert algo(strategy)

    total_return = strategy.temp["total_return"]
    assert total_return["A"] == pytest.approx(1.0)
    assert total_return["B"] == pytest.approx(-1.0)
    stats_row = algo.stats.loc[prices.index[1]]
    assert stats_row["MEAN"] == pytest.approx(0.0)


def test_total_return_one_row_window_returns_zero():
    prices = _prices(A=[100.0, 110.0])
    strategy = _strategy_context(prices, now_idx=0)

    assert TotalReturn(lookback=pd.DateOffset(days=5))(strategy)
    assert strategy.temp["total_return"]["A"] == pytest.approx(0.0)


def test_total_return_validates_standardize_flag():
    with pytest.raises(TypeError, match="standardize"):
        TotalReturn(standardize="yes")  # type: ignore[arg-type]


def test_backtest_summary_plot_factor_stats_builds_expected_lines(monkeypatch):
    called = {"show": 0}

    def fake_show(self):  # noqa: ANN001
        called["show"] += 1

    monkeypatch.setattr("visualization.figure.Figure.show", fake_show)

    total_return_stats = pd.DataFrame(
        {
            "TOTAL_COVERED": [2, 2],
            "MEAN": [0.1, 0.2],
            "MEDIAN": [0.08, 0.18],
            "25TH": [0.02, 0.04],
            "75TH": [0.12, 0.24],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    signal_algo = SimpleNamespace(factor_stats={"total_return": total_return_stats})

    backtest = SimpleNamespace(
        name="Top",
        strategy=SimpleNamespace(
            prices=pd.Series(
                [100.0, 101.0],
                index=pd.date_range("2024-01-01", periods=2, freq="D"),
                name="Top",
            ),
            algos={"MomentumSignal": signal_algo},
            data=pd.DataFrame(),
            universe=pd.DataFrame(),
            outlays=pd.DataFrame(),
        ),
        weights=pd.DataFrame(),
        security_weights=pd.DataFrame(),
        positions=pd.DataFrame(),
        get_transactions=pd.DataFrame(),
    )
    summary = BacktestSummary(backtest)

    fig = summary.plot_factor_stats("MomentumSignal", "total_return", "Top")
    built = fig.build().fig

    assert built is not None
    assert built.layout.title.text == "Factor Stats - MomentumSignal Total Return - Top"
    assert {trace.name for trace in built.data} == {"MEAN", "MEDIAN", "25TH", "75TH"}
    assert built.data[2].line.dash == "dot"
    assert built.data[3].line.dash == "dot"
    assert called["show"] == 1
