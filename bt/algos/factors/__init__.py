"""Canonical factor exports."""

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
