from typing import Any

from bt.core.algo_base import Algo
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
        if self._remaining_days > 0:
            self._remaining_days -= 1
            return False
        return True
