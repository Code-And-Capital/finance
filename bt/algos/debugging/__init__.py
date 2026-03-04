"""Debugging and observability algorithms for the backtesting engine.

This package provides non-trading helper algos used during strategy
development and troubleshooting, including structured logging helpers and
an interactive debugger trigger.
"""

# Debugging / inspection algos
from .prints import PrintDate, PrintTempData, PrintInfo
from .debug import Debug

# Expose all classes in package namespace
__all__ = [
    "PrintDate",
    "PrintTempData",
    "PrintInfo",
    "Debug",
]
