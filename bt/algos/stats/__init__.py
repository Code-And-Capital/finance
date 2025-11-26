"""
Statistical algorithms for the bt framework.

This subpackage contains algorithms that compute, store, or manipulate
metrics used for selection, ranking, or weighting of assets. These
algos generally populate temp['stat'] or provide other calculated
statistics that downstream algos rely on.
"""

from .returns import StatTotalReturn
from .risk import UpdateRisk
from .set_stat import SetStat

__all__ = [
    "StatTotalReturn",
    "UpdateRisk",
    "SetStat",
]
