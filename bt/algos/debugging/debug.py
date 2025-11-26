from bt.core import Algo
from typing import Optional, Callable


class Debug(Algo):
    """
    Algo that invokes a Python debugger breakpoint (`pdb.set_trace()`), allowing
    interactive inspection of the backtest environment.

    This is intended purely for debugging. When triggered, execution will pause
    and open an interactive debugging session where you can inspect variables,
    step through code, or examine the ``target`` object through its StrategyBase
    interface.

    Examples
    --------
    Always trigger debugging:
        Debug()

    Trigger only when a condition is met:
        Debug(condition=lambda t: t.now.day == 1)

    Attributes
    ----------
    condition : callable or None
        Optional function receiving the target. If provided, debugging will only
        activate when ``condition(target)`` evaluates to True.
    """

    def __init__(self, condition: Optional[Callable] = None):
        """
        Initialize the Debug algo.

        Parameters
        ----------
        condition : callable, optional
            A function that accepts the target object and returns True when the
            debugger should activate. If None, debugging is always triggered.
        """
        super().__init__()
        self.condition = condition

    def __call__(self, target) -> bool:
        """
        Invoke `pdb.set_trace()` to start a debugging session.

        Parameters
        ----------
        target : bt.backtest.Target
            The backtest context object available inside the debugging session.
            You can inspect its attributes, temporary data, algo chain state,
            and portfolio information.

        Returns
        -------
        bool
            Always returns True so the algo chain continues after exiting the
            debugger.
        """
        import pdb

        should_break = True
        if self.condition is not None:
            try:
                should_break = bool(self.condition(target))
            except Exception as exc:
                print(f"[Debug] Condition function raised an exception: {exc}")
                should_break = False

        if should_break:
            pdb.set_trace()

        return True
