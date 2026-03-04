from typing import Any

from bt.core.algo_base import Algo
from utils.math_utils import validate_integer, validate_non_negative


class RunEveryNPeriods(Algo):
    """Trigger execution every ``n`` unique periods.

    Parameters
    ----------
    n : int
        Positive period interval between triggers.
    offset : int, optional
        Initial phase offset in ``[0, n - 1]``.

    Notes
    -----
    The algo uses a cyclic phase counter in ``[0, n-1]`` and triggers when
    ``phase == offset``. It suppresses duplicate triggers when called multiple
    times for the same ``target.now``.
    """

    def __init__(self, n: int, offset: int = 0):
        """Initialize periodic trigger.

        Parameters
        ----------
        n : int
            Positive interval between trigger events.
        offset : int, optional
            Initial trigger phase in ``[0, n - 1]``.

        Raises
        ------
        TypeError
            If ``n`` or ``offset`` is not an integer.
        ValueError
            If ``n <= 0`` or offset is outside ``[0, n - 1]``.
        """
        super().__init__()
        n_val = int(
            validate_non_negative(
                validate_integer(n, "RunEveryNPeriods `n`"),
                "RunEveryNPeriods `n`",
            )
        )
        offset_val = int(
            validate_non_negative(
                validate_integer(offset, "RunEveryNPeriods `offset`"),
                "RunEveryNPeriods `offset`",
            )
        )
        if n_val == 0:
            raise ValueError("RunEveryNPeriods `n` must be > 0.")
        if offset_val >= n_val:
            raise ValueError("RunEveryNPeriods `offset` must satisfy 0 <= offset < n.")

        self.n = n_val
        self.offset = offset_val
        self._phase = 0
        self._last_call = None

    def __call__(self, target: Any) -> bool:
        """Evaluate whether the current call should trigger.

        Parameters
        ----------
        target : Any
            Target object expected to expose ``now``.

        Returns
        -------
        bool
            ``True`` when current period is a trigger boundary, else ``False``.
        """
        now = getattr(target, "now", None)
        if now is None:
            return False

        if self._last_call == now:
            return False

        self._last_call = now

        should_run = self._phase == self.offset
        self._phase = (self._phase + 1) % self.n
        return should_run
