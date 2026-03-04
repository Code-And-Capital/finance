"""
Flow control algos for bt package.

This package contains scheduling and control-flow algos that determine
when a strategy should execute. Examples include running once, running
on specific dates, periodic execution, and rebalancing triggers.

Modules:
---------
- combinators.py
- control.py
- close_positions.py
- run_once.py
- run_period.py
- run_on_date.py
- run_after.py
- run_every_n_periods.py
- run_if_out_of_bounds.py
"""

# flow control
from .control import Require

# combinators
from .combinators import Or, Not

# closer
from .close_positions import ClosePositionsAfterDates

# Base / simple algos
from .run_once import RunOnce

# Periodic base
from .run_period import RunPeriod
from .run_period import RunDaily
from .run_period import RunWeekly
from .run_period import RunMonthly
from .run_period import RunQuarterly
from .run_period import RunYearly

# Date-specific triggers
from .run_on_date import RunOnDate
from .run_after import RunAfterDate, RunAfterDays, RunAfterMonths
from .run_every_n_periods import RunEveryNPeriods, RunEveryNMonths

# Rebalancing / bounds triggers
from .run_if_out_of_bounds import RunIfOutOfBounds, RunIfCashOutOfBounds

# Expose all classes in package namespace
__all__ = [
    "Require",
    "Or",
    "Not",
    "ClosePositionsAfterDates",
    "RunOnce",
    "RunPeriod",
    "RunDaily",
    "RunWeekly",
    "RunMonthly",
    "RunQuarterly",
    "RunYearly",
    "RunOnDate",
    "RunAfterDate",
    "RunAfterDays",
    "RunAfterMonths",
    "RunEveryNPeriods",
    "RunEveryNMonths",
    "RunIfOutOfBounds",
    "RunIfCashOutOfBounds",
]
