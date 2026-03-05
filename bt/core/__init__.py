"""
Core module for the portfolio/strategy framework.

This package exposes the foundational building blocks used throughout the
system, including:

- Node / Strategy tree structures
- Base classes for algorithm components
- Security objects and related utilities
"""

# ---------------------------------------------------------------------
# Node & Strategy System
# ---------------------------------------------------------------------
from .nodes import Node
from .strategy import Strategy, StrategyBase

# ---------------------------------------------------------------------
# Security / Asset Representation
# ---------------------------------------------------------------------
from .security import (
    Security,
    SecurityBase,
    FixedIncomeSecurity,
    HedgeSecurity,
    CouponPayingSecurity,
    CouponPayingHedgeSecurity,
)

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    "Node",
    "Strategy",
    "StrategyBase",
    "Security",
    "SecurityBase",
    "FixedIncomeSecurity",
    "HedgeSecurity",
    "CouponPayingSecurity",
    "CouponPayingHedgeSecurity",
]
