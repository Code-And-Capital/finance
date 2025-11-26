"""
Capital adjustment algorithms for the bt framework.

This subpackage contains algorithms that model capital flows, inflows,
outflows, or other adjustments to a strategy's capital that do not
directly affect returns. These algos typically call target.adjust(amount)
and are useful for modeling contributions, withdrawals, or other
portfolio-level adjustments.
"""

from .capital_flow import CapitalFlow
from .margin import Margin

__all__ = [
    "CapitalFlow",
    "Margin",
]
