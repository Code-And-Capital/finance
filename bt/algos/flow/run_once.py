from bt.core import Algo
from typing import Optional


class RunOnce(Algo):
    """
    Algo that only returns ``True`` on its first invocation. On all subsequent
    calls, it returns ``False``.

    This is useful in scenarios where an action should only be executed once
    during the entire backtest—for example, performing an initial buy-and-hold
    trade or initializing strategy state.

    Examples
    --------
    Always run once:
        RunOnce()

    Conditionally run once:
        RunOnce(run_on_first_call=False)

    Attributes
    ----------
    run_on_first_call : bool
        Whether the algorithm should run on the very first call.
    has_run : bool
        Tracks whether the algo has already executed.
    """

    def __init__(self, run_on_first_call: bool = True):
        """
        Initialize the RunOnce algo.

        Parameters
        ----------
        run_on_first_call : bool, optional
            If True (default), the algo returns True on its first call.
            If False, the first call will also return False, and the algo
            will never run.
        """
        super().__init__()
        self.run_on_first_call = run_on_first_call
        self.has_run = False

    def __call__(self, target) -> bool:
        """
        Execute the algo.

        Returns ``True`` only on the first call (depending on configuration),
        and returns ``False`` on all further calls.

        Parameters
        ----------
        target : bt.backtest.Target
            Backtest context object (unused, but required by algo signature).

        Returns
        -------
        bool
            True if the algo is executing for the first time and allowed to
            run. False otherwise.
        """
        # First call
        if not self.has_run:
            self.has_run = True
            return self.run_on_first_call

        # All subsequent calls
        return False
