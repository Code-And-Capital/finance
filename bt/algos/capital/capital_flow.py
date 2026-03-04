from typing import Any

from bt.core.algo_base import Algo


class CapitalFlow(Algo):
    """Apply a capital flow to a strategy-like target.

    This algorithm models external capital movements such as contributions
    (positive `amount`) and withdrawals (negative `amount`). The adjustment
    is applied through the target's ``adjust`` method and is treated according
    to that method's default flow semantics.

    Parameters
    ----------
    amount : float
        Capital amount to apply. Positive values add capital, negative values
        remove capital.
    """

    def __init__(self, amount: float) -> None:
        """Initialize the capital flow algorithm.

        Parameters
        ----------
        amount : float
            Capital amount to apply on each invocation.
        """
        super().__init__()
        self.amount = float(amount)

    def __call__(self, target: Any) -> bool:
        """Execute the capital flow.

        Parameters
        ----------
        target : Any
            Strategy-like object that must implement ``adjust(amount)``.

        Returns
        -------
        bool
            Always ``True`` so execution in an algo stack can continue.

        Raises
        ------
        AttributeError
            If ``target`` does not implement an ``adjust`` method.
        """
        if not hasattr(target, "adjust"):
            raise AttributeError(
                "CapitalFlow target must implement an `adjust(amount)` method."
            )
        target.adjust(self.amount)
        return True
