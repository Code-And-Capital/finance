"""
Performance benchmarks
"""

import sys
import os

import numpy as np
import pandas as pd
import cProfile
from bt.core import Strategy, Security
from bt.algos.selection import SelectRandomly, SelectThese
from bt.algos.weighting import WeighRandomly
from bt.algos.flow import RunMonthly
from bt.algos.portfolio_ops import Rebalance
from bt.engine import Backtest
import bt


def benchmark_1():
    x = np.random.randn(10000, 1000) * 0.01
    idx = pd.date_range("1990-01-01", freq="B", periods=x.shape[0])
    data = np.exp(pd.DataFrame(x, index=idx).cumsum())

    s = Strategy(
        "s",
        [
            RunMonthly(),
            SelectRandomly(len(data.columns) / 2),
            WeighRandomly(),
            Rebalance(),
        ],
    )

    t = Backtest(s, data)
    return bt.run(t)


def benchmark_3():
    # Similar to benchmark_1, but with trading in only a small subset of assets
    # However, because the "multipier" is used, we can't just pass the string
    # names to the constructor, and so the solution is to use the lazy_add flag.
    # Changing lazy_add to False demonstrates the performance gain.
    # i.e. on Win32, it went from 4.3s with the flag to 10.9s without.

    x = np.random.randn(10000, 1000) * 0.01
    idx = pd.date_range("1990-01-01", freq="B", periods=x.shape[0])
    data = np.exp(pd.DataFrame(x, index=idx).cumsum())
    children = [Security(name=i, multiplier=10, lazy_add=False) for i in range(1000)]
    s = Strategy(
        "s",
        [
            RunMonthly(),
            SelectThese([0, 1]),
            WeighRandomly(),
            Rebalance(),
        ],
        children=children,
    )

    t = Backtest(s, data)
    return bt.run(t)


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
