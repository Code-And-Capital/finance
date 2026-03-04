"""
Selection algorithms for the bt framework.

This subpackage contains all algorithms that operate on the selection
phase of the strategy pipeline. They populate or filter the list in
temp['selected'] so that later statistics or weighting algos know
which tickers to work on.

Included tools range from simple selection (SelectAll, SelectThese)
to more advanced filters (SelectHasData), random
selection for benchmarking, and logical/boolean selection utilities.
"""

from .base_selection import SelectAll, SelectHasData, SelectActive
from .these import SelectThese, SelectWhere
from .rank import SelectN
from .random import SelectRandomly

__all__ = [
    "SelectAll",
    "SelectThese",
    "SelectHasData",
    "SelectN",
    "SelectRandomly",
    "SelectWhere",
    "SelectActive",
]
