import math
from typing import Any

from bt.core.algo_base import Algo


class Margin(Algo):
    """Model margin borrowing costs and maintenance checks.

    The algorithm interprets negative strategy capital as borrowed margin,
    accrues compound interest between invocations, applies that interest as a
    non-flow fee, and enforces a maintenance threshold by triggering
    liquidation when equity falls below ``requirement`` of gross portfolio
    value.

    Parameters
    ----------
    rate : float
        Annualized margin interest rate (e.g., ``0.05`` for 5%).
    requirement : float
        Maintenance equity requirement as a fraction in ``(0, 1]``.
    """

    def __init__(self, rate: float, requirement: float) -> None:
        """Initialize the margin model.

        Parameters
        ----------
        rate : float
            Annualized margin interest rate.
        requirement : float
            Maintenance requirement as an equity ratio threshold.

        Raises
        ------
        ValueError
            If ``rate <= -1`` or ``requirement`` is outside ``(0, 1]``.
        """
        super().__init__(Margin.__name__)
        if rate <= -1.0:
            raise ValueError("Margin `rate` must be greater than -1.0.")
        if not (0.0 < requirement <= 1.0):
            raise ValueError("Margin `requirement` must be in the range (0, 1].")

        self.rate = float(rate)
        self.requirement = float(requirement)
        self._last_date = None
        self._daily_growth_factor = math.pow(1.0 + self.rate, 1.0 / 365.25)

    def _daily_rate(self) -> float:
        """Return the equivalent daily compounded rate.

        Returns
        -------
        float
            Effective one-day interest rate.
        """
        return self._daily_growth_factor - 1.0

    def _interest_factor(self, elapsed_days: float) -> float:
        """Compute compound interest factor for an elapsed time interval.

        Parameters
        ----------
        elapsed_days : float
            Elapsed time in calendar days. Fractional values are supported.

        Returns
        -------
        float
            Compound factor minus one over ``elapsed_days``.
        """
        return math.pow(self._daily_growth_factor, elapsed_days) - 1.0

    def __call__(self, target: Any) -> bool:
        """Apply accrued margin interest and maintenance enforcement.

        Parameters
        ----------
        target : Any
            Strategy-like object that must expose:
            ``now``, ``capital``, ``children``, ``value``, ``adjust``, and
            ``allocate``.

        Returns
        -------
        bool
            Always ``True`` so execution in an algo stack can continue.

        Raises
        ------
        AttributeError
            If required target attributes/methods are missing.
        ValueError
            If ``target.now`` is earlier than the previous invocation time.

        Notes
        -----
        Interest charges are applied with ``flow=False`` so financing costs
        affect performance rather than being treated as external cash flows.
        """
        required_attrs = ("now", "capital", "children", "value", "adjust", "allocate")
        missing = [attr for attr in required_attrs if not hasattr(target, attr)]
        if missing:
            raise AttributeError(
                "Margin target missing required attributes/methods: "
                + ", ".join(sorted(missing))
            )

        if self._last_date is None:
            self._last_date = target.now
            return True

        # Number of days since last interest calculation (supports intraday timing).
        diff = target.now - self._last_date
        elapsed_days = diff.total_seconds() / 86_400.0
        if elapsed_days < 0:
            raise ValueError(
                "Margin received non-monotonic dates: `target.now` moved backwards."
            )

        # Margin amount is negative capital
        margin = -target.capital

        if margin > 0:
            # Total portfolio value
            port_val = sum(child.value for child in target.children.values())

            # Interest on the margin
            fee = margin * self._interest_factor(elapsed_days)

            # Charge interest
            target.adjust(-fee, flow=False, fee=fee)

            # Check equity ratio against requirement
            if port_val > 0:
                equity_ratio = target.value / port_val
            else:
                equity_ratio = math.inf

            if port_val > 0 and equity_ratio < self.requirement:
                max_value = target.value * (1 / self.requirement)

                # Liquidate to cover shortfall
                deficit = max_value - port_val
                target.allocate(deficit / 2)

        # Update last date
        self._last_date = target.now
        return True
