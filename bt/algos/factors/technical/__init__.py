"""Technical-analysis algorithms for the bt framework.

This subpackage contains algorithms that compute indicator-like analytics used
for selection, ranking, and portfolio construction. Most algos in this module
write cross-sectional values into ``target.temp`` (for example
``target.temp["stat"]`` or moving-average vectors) for downstream consumption.
"""

from .returns import TotalReturn
from .moving_average import (
    ExponentialWeightedMovingAverage,
    KernelMovingAverage,
    SimpleMovingAverage,
)

__all__ = [
    "TotalReturn",
    "SimpleMovingAverage",
    "ExponentialWeightedMovingAverage",
    "KernelMovingAverage",
]
