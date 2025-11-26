from bt.core import Algo


class CapitalFlow(Algo):
    """
    Models capital flows (inflows or outflows) for a target node.

    This Algo adjusts the capital of a strategy node without affecting
    returns. It can be used to simulate inflows such as contributions
    or outflows such as withdrawals. The adjustment remains in the
    strategy until the next reallocation or rebalance.

    Args:
        amount (float): Amount of capital to adjust (positive for inflow, negative for outflow).
    """

    def __init__(self, amount: float) -> None:
        """
        Initialize a CapitalFlow instance.

        Parameters:
            amount (float): Amount of capital to adjust.
        """
        super().__init__()
        self.amount = float(amount)

    def __call__(self, target) -> bool:
        """
        Apply the capital adjustment to the target node.

        Parameters:
            target: Strategy/backtest node supporting 'adjust(amount)'.

        Returns:
            bool: Always True to indicate successful application.
        """
        target.adjust(self.amount)
        return True
