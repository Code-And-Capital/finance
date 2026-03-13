"""
Factors for the bt framework.

Factors are responsible for calculating a numeric value per
security so strategies can rank, filter, compare, or otherwise
act on cross-sectional measurements.
"""

from .core import Factor, SetFactor
from .technical import (
    ExponentialWeightedMovingAverage,
    KernelMovingAverage,
    SimpleMovingAverage,
    TotalReturn,
)

__all__ = [
    "Factor",
    "SetFactor",
    "TotalReturn",
    "SimpleMovingAverage",
    "ExponentialWeightedMovingAverage",
    "KernelMovingAverage",
]
