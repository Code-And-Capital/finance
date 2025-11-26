from typing import Any
from bt.core.algo_base import Algo


class RunEveryNPeriods(Algo):
    """
    Algo that triggers every `n` periods.

    Useful for strategies that need to execute periodically, e.g.,
    rebalancing every month, or performing actions every n trading periods.

    Args:
        n : int
            Number of periods between executions.
        offset : int, optional
            Offset for the first execution. If 0 (default), the algo will run
            the first time it is called. Use a different offset to stagger
            multiple strategies.

    Examples
    --------
    Run every 3 periods, starting immediately:
        RunEveryNPeriods(3)

    Run every 3 periods, but start on the second period:
        RunEveryNPeriods(3, offset=1)
    """

    def __init__(self, n: int, offset: int = 0):
        """
        Initialize RunEveryNPeriods.

        Parameters
        ----------
        n : int
            Number of periods between executions.
        offset : int, optional
            Offset for first execution. Defaults to 0.
        """
        super().__init__()
        self.n: int = n
        self.offset: int = offset
        self.idx: int = n - offset - 1  # internal counter
        self.last_call = None  # tracks last period to avoid multiple calls

    def __call__(self, target: Any) -> bool:
        """
        Determine whether the algo should run in the current period.

        Parameters
        ----------
        target : bt.backtest.Target
            The backtest target containing the `now` attribute (current date).

        Returns
        -------
        bool
            True if the algo should execute this period, False otherwise.
        """
        # prevent multiple triggers on the same period
        if self.last_call == target.now:
            return False

        self.last_call = target.now

        if self.idx == self.n - 1:
            # reset counter after triggering
            self.idx = 0
            return True
        else:
            self.idx += 1
            return False
