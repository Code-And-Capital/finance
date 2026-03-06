"""
Signals for the bt framework.

This subpackage contains signal building blocks used
within the strategy pipeline. Components are responsible
for generating directional or actionable insights that
inform downstream components such as selection, sizing,
or execution. They may be derived from technical indicators
(e.g., momentum, trend, volatility), fundamental data (e.g.,
valuation, earnings, macro inputs), or hybrid approaches. Each
component encapsulates a specific hypothesis about market behavior
and exposes a standardized interface so strategies can combine,
compare, or aggregate signals in a consistent way.
"""

from .core import Signal
from .trend import PriceCrossOverSignal, MomentumSignal, DualMACrossoverSignal

__all__ = [
    "Signal",
    "PriceCrossOverSignal",
    "MomentumSignal",
    "DualMACrossoverSignal",
]
