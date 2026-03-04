from typing import Any, Callable

from bt.core.algo_base import Algo
import utils.logging as logging


class Debug(Algo):
    """Trigger an interactive debugger breakpoint during algo execution.

    Parameters
    ----------
    condition : Callable[[Any], bool] | None, optional
        Optional predicate evaluated against the target. If provided, the
        debugger is activated only when the predicate returns ``True``.

    Notes
    -----
    This utility is intended for local development and diagnostics only.
    """

    def __init__(self, condition: Callable[[Any], bool] | None = None):
        """Initialize the conditional debugger trigger."""
        super().__init__()
        self.condition = condition

    def __call__(self, target: Any) -> bool:
        """Evaluate condition and optionally invoke ``pdb.set_trace()``.

        Parameters
        ----------
        target : Any
            Target object passed by the algo stack.

        Returns
        -------
        bool
            Always ``True`` so the algo stack continues.
        """
        import pdb

        should_break = True
        if self.condition is not None:
            try:
                should_break = bool(self.condition(target))
            except Exception as exc:
                logging.log(
                    f"[Debug] Condition function raised an exception: {exc}",
                    type="warning",
                )
                should_break = False

        if should_break:
            pdb.set_trace()

        return True
