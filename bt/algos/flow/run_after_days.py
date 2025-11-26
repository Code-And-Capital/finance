from typing import Any
from bt.core.algo_base import Algo


class RunAfterDays(Algo):
    """
    Algo that triggers only after a specified number of trading days have passed.

    This is useful for strategies that require an initial warm-up period,
    such as calculating trailing averages or waiting until sufficient
    data has accumulated before taking action.

    Args:
        days : int
            The number of trading days to wait before the algo starts executing.

    Examples
    --------
    Wait for 5 trading days before executing:
        RunAfterDays(5)
    """

    def __init__(self, days: int):
        """
        Initialize RunAfterDays with a warm-up period.

        Parameters
        ----------
        days : int
            Number of trading days to wait before starting execution.
        """
        super().__init__()
        self.remaining_days: int = days

    def __call__(self, target: Any) -> bool:
        """
        Execute the algo and decrement the remaining warm-up days.

        Parameters
        ----------
        target : bt.backtest.Target
            The backtest target containing the `now` attribute (current date).
            This argument is unused in this algo but required by the Algo signature.

        Returns
        -------
        bool
            False if the warm-up period is still ongoing, True once the
            specified number of trading days have passed.
        """
        if self.remaining_days > 0:
            self.remaining_days -= 1
            return False
        return True
