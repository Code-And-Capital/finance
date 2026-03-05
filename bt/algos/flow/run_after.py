from typing import Any

import pandas as pd

from bt.algos.core import Algo
from utils.date_utils import coerce_timestamp, month_index
from utils.math_utils import validate_integer, validate_non_negative


class RunAfterDays(Algo):
    """Gate execution until a configured number of invocations has elapsed.

    Parameters
    ----------
    days : int
        Number of calls to return ``False`` before returning ``True``.
        Must be a non-negative integer.

    Notes
    -----
    This algo counts invocations, not calendar days.
    """

    def __init__(self, days: int):
        """Initialize invocation warm-up gate.

        Parameters
        ----------
        days : int
            Number of calls to delay execution.

        Raises
        ------
        TypeError
            If ``days`` is not an integer value.
        ValueError
            If ``days`` is negative.
        """
        super().__init__()
        days_val = validate_integer(days, "RunAfterDays `days`")
        self._remaining_days = int(
            validate_non_negative(days_val, "RunAfterDays `days`")
        )

    def __call__(self, target: Any) -> bool:
        """Evaluate whether warm-up is complete.

        Parameters
        ----------
        target : Any
            Unused target argument required by the algo interface.

        Returns
        -------
        bool
            ``False`` during warm-up, otherwise ``True``.
        """
        if self._remaining_days == 0:
            return True

        self._remaining_days -= 1
        return False


class RunAfterDate(Algo):
    """Return ``True`` only when current date is strictly after a threshold.

    Parameters
    ----------
    date : Any
        Date-like threshold value parsed by :func:`pandas.to_datetime`.

    Notes
    -----
    - Strictly greater-than: when ``target.now == date``, result is ``False``.
    - Once it returns ``True`` for the first time, it remains ``True``.
    """

    def __init__(self, date: Any):
        """Initialize date threshold gate.

        Parameters
        ----------
        date : Any
            Date-like value converted to ``pandas.Timestamp``.

        Raises
        ------
        ValueError
            If ``date`` cannot be parsed into a valid timestamp.
        """
        super().__init__()
        self.date: pd.Timestamp = coerce_timestamp(date, "RunAfterDate `date`")
        self._is_active = False

    def __call__(self, target: Any) -> bool:
        """Evaluate whether ``target.now`` is strictly after configured date."""
        if self._is_active:
            return True

        now = self._resolve_now(target)
        if now is None:
            return False

        if now > self.date:
            self._is_active = True
            return True
        return False


class RunAfterMonths(Algo):
    """Gate execution until a configured number of calendar months has elapsed.

    Parameters
    ----------
    months : int
        Number of calendar months to wait from the first observed ``target.now``.
        Must be a non-negative integer.

    Notes
    -----
    - Month elapsed uses ``year * 12 + month`` indexing.
    - Invalid or missing ``target.now`` returns ``False``.
    """

    def __init__(self, months: int):
        """Initialize month warm-up gate.

        Parameters
        ----------
        months : int
            Number of calendar months to delay execution.

        Raises
        ------
        TypeError
            If ``months`` is not an integer value.
        ValueError
            If ``months`` is negative.
        """
        super().__init__()
        months_val = validate_integer(months, "RunAfterMonths `months`")
        self._months = int(validate_non_negative(months_val, "RunAfterMonths `months`"))
        self._first_date: pd.Timestamp | None = None
        self._is_active = self._months == 0

    def __call__(self, target: Any) -> bool:
        """Evaluate whether month warm-up is complete."""
        if self._is_active:
            return True

        now = self._resolve_now(target)
        if now is None:
            return False

        if self._first_date is None:
            self._first_date = now

        elapsed_months = month_index(now) - month_index(self._first_date)
        if elapsed_months >= self._months:
            self._is_active = True
            return True
        return False
