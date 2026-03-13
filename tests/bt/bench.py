"""
Performance benchmarks
"""

import cProfile

import numpy as np
import pandas as pd

from bt.algos.flow import RunMonthly
from bt.algos.portfolio_ops import Rebalance
from bt.algos.selection import SelectRandomly, SelectThese
from bt.algos.weighting import WeightRandomly
from bt.analytics import BacktestSummary
from bt.core import Backtest
from bt.core import Security, Strategy


def _run_summary(backtest: Backtest) -> BacktestSummary:
    backtest.run()
    return BacktestSummary(backtest)


def benchmark_1():
    x = np.random.randn(10000, 1000) * 0.01
    idx = pd.date_range("1990-01-01", freq="B", periods=x.shape[0])
    data = np.exp(pd.DataFrame(x, index=idx).cumsum())

    s = Strategy(
        "s",
        [
            RunMonthly(),
            SelectRandomly(int(len(data.columns) / 2)),
            WeightRandomly(),
            Rebalance(),
        ],
    )

    t = Backtest(s, data)
    return _run_summary(t)


def benchmark_3():
    x = np.random.randn(10000, 1000) * 0.01
    idx = pd.date_range("1990-01-01", freq="B", periods=x.shape[0])
    data = np.exp(pd.DataFrame(x, index=idx).cumsum())
    data.columns = data.columns.map(str)
    children = [Security(name=column) for column in data.columns]
    s = Strategy(
        "s",
        [
            RunMonthly(),
            SelectThese(["0", "1"]),
            WeightRandomly(),
            Rebalance(),
        ],
        children=children,
    )

    t = Backtest(s, data)
    return _run_summary(t)


if __name__ == "__main__":
    print("\n\n\n================= Benchmark 1 =======================\n")
    cProfile.run("benchmark_1()", sort="tottime")
    print("\n----------------- Benchmark 1 -----------------------\n\n\n")

    print("\n\n\n================= Benchmark 2 =======================\n")
    cProfile.run("benchmark_2()", sort="tottime")
    print("\n----------------- Benchmark 2 -----------------------\n\n\n")

    print("\n\n\n================= Benchmark 3 =======================\n")
    cProfile.run("benchmark_3()", sort="cumtime")
    print("\n----------------- Benchmark 3 -----------------------\n\n\n")
