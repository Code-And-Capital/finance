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
from .backtest import Backtest
from .strategy import Strategy

# ---------------------------------------------------------------------
# Security / Asset Representation
# ---------------------------------------------------------------------
from .security import (
    Security,
)
from .commission import (
    zero_commission,
    quantity_tiered_commission,
    notional_bps_commission,
    fixed_per_trade_commission,
    per_share_commission,
    per_share_with_min_max_commission,
    notional_bps_with_min_commission,
    tiered_notional_bps_commission,
    sec_finra_sell_fee,
)

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    "Node",
    "Backtest",
    "Strategy",
    "Security",
    "zero_commission",
    "quantity_tiered_commission",
    "notional_bps_commission",
    "fixed_per_trade_commission",
    "per_share_commission",
    "per_share_with_min_max_commission",
    "notional_bps_with_min_commission",
    "tiered_notional_bps_commission",
    "sec_finra_sell_fee",
]
