from typing import Any

from bt.core.algo_base import Algo


class RunOnce(Algo):
    """Return ``True`` at most once across the algo instance lifecycle.

    Parameters
    ----------
    run_on_first_call : bool, optional
        If ``True`` (default), the first invocation returns ``True``.
        If ``False``, all invocations return ``False``.

    Notes
    -----
    State is instance-local. Reusing the same instance across backtests
    preserves execution state.
    If ``target.inow == 0`` (initial synthetic/bootstrapping row), this algo
    returns ``False`` and does not consume its one-shot run.
    """

    def __init__(self, run_on_first_call: bool = True):
        """Initialize the one-shot execution gate."""
        super().__init__()
        self.run_on_first_call = bool(run_on_first_call)
        self._has_run = False

    def __call__(self, target: Any) -> bool:
        """Evaluate one-shot run eligibility.

        Parameters
        ----------
        target : Any
            Unused target parameter required by the algo interface.

        Returns
        -------
        bool
            ``True`` only when the first invocation is permitted by
            ``run_on_first_call``; otherwise ``False``.
        """
        if getattr(target, "inow", None) == 0:
            return False

        if not self._has_run:
            self._has_run = True
            return self.run_on_first_call

        return False
