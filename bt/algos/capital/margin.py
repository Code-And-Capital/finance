import math
from bt.core.algo_base import Algo


class Margin(Algo):
    """
    Models margin lending in a strategy.

    This algorithm charges a periodic interest on borrowed capital (margin)
    and will liquidate positions if the equity falls below a specified
    maintenance requirement.

    Args:
        rate (float): The margin interest rate (e.g., 0.05 for 5% annual).
        requirement (float): Maintenance requirement. The strategy will
            liquidate positions if equity falls below this fraction of portfolio value.
    """

    def __init__(self, rate: float, requirement: float) -> None:
        """
        Initialize the Margin algorithm.

        Parameters:
            rate (float): Annual interest rate on margin.
            requirement (float): Maintenance requirement (fraction of portfolio).
        """
        super().__init__(Margin.__name__)
        self.rate = rate
        self.requirement = requirement
        self._last_date = None

    def _daily_rate(self) -> float:
        """
        Compute daily interest rate from annual rate.

        Returns:
            float: Daily interest rate.
        """
        return math.pow(1 + self.rate, 1 / 365.25) - 1

    def __call__(self, target) -> bool:
        """
        Apply margin interest and enforce maintenance requirement.

        Parameters:
            target: Strategy/backtest node with:
                - capital: current borrowed margin (negative if borrowed)
                - children: dict of positions with .value attribute
                - value: total equity of the strategy
                - adjust(amount, fee=None): method to adjust capital
                - allocate(amount): method to allocate/liquidate funds
                - now: current date

        Returns:
            bool: Always True
        """
        if self._last_date is None:
            self._last_date = target.now
            return True

        # Number of days since last interest calculation
        diff = target.now - self._last_date

        # Margin amount is negative capital
        margin = -target.capital

        if margin > 0:
            # Total portfolio value
            port_val = sum(child.value for child in target.children.values())

            # Interest on the margin
            f = math.pow(1 + self._daily_rate(), diff.days) - 1
            fee = margin * f

            # Charge interest
            target.adjust(-fee, fee=fee)

            # Check equity ratio against requirement
            equity_ratio = target.value / port_val
            if equity_ratio < self.requirement:
                max_value = target.value * (1 / self.requirement)

                # Liquidate to cover shortfall
                deficit = max_value - port_val
                target.allocate(deficit / 2)

        # Update last date
        self._last_date = target.now
        return True
