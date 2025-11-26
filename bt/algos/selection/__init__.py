"""
Selection algorithms for the bt framework.

This subpackage contains all algorithms that operate on the selection
phase of the strategy pipeline. They populate or filter the list in
temp['selected'] so that later statistics or weighting algos know
which tickers to work on.

Included tools range from simple selection (SelectAll, SelectThese)
to more advanced filters (SelectHasData, SelectMomentum), random
selection for benchmarking, and logical/boolean selection utilities.
"""

from .all import SelectAll
from .these import SelectThese, SelectWhere
from .filter import SelectHasData, SelectActive
from .rank import SelectN
from .momentum import SelectMomentum
from .random import SelectRandomly
from .regex import SelectRegex

__all__ = [
    "SelectAll",
    "SelectThese",
    "SelectHasData",
    "SelectN",
    "SelectMomentum",
    "StatTotalReturn",
    "SelectRandomly",
    "SelectWhere",
    "SelectRegex",
    "SelectActive",
]
