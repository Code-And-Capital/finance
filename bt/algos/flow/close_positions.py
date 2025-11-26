from bt.core import Algo, SecurityBase


class ClosePositionsAfterDates(Algo):
    """
    Close positions on securities after specified dates.

    This algorithm ensures that positions on matured, redeemed, or
    time-limited securities are closed after a given date. It can be used
    to enforce rules such as "do not hold securities with time to maturity
    less than one year."

    Notes:
        - If placed after a RunPeriod algo in the stack, actual closing will
          occur after the provided date. The price of the security must exist
          up to that date, or use the @run_always decorator to close immediately.
        - Does not operate via temp['weights'] and Rebalance, so hedges and
          other special securities are also properly closed.

    Args:
        close_dates (str): Name of a DataFrame (indexed by security name) with a
            "date" column indicating the date after which positions should be closed.

    Sets:
        target.perm['closed']: Set of securities already closed.
    """

    def __init__(self, close_dates: str) -> None:
        """
        Initialize ClosePositionsAfterDates.

        Parameters:
            close_dates (str): Name of the DataFrame with closing dates.
        """
        super().__init__()
        self.close_dates = close_dates

    def __call__(self, target) -> bool:
        """
        Close positions for securities whose closing dates have passed.

        Steps:
        1. Initialize target.perm['closed'] if not already present.
        2. Fetch the close_dates DataFrame via target.get_data.
        3. Identify securities eligible for closing.
        4. Close positions and mark them as closed.
        5. Update the target's root.

        Parameters:
            target: Strategy/backtest object providing:
                - children: dict of current positions
                - perm: dict for persistent info
                - get_data(name): fetch DataFrame
                - close(sec_name, update=False): method to close a position
                - root.update(now): propagate updates

        Returns:
            bool: Always True
        """
        if "closed" not in target.perm:
            target.perm["closed"] = set()

        close_dates_df = target.get_data(self.close_dates)["date"]

        # Candidate securities for closing
        sec_names = [
            sec_name
            for sec_name, sec in target.children.items()
            if isinstance(sec, SecurityBase)
            and sec_name in close_dates_df.index
            and sec_name not in target.perm["closed"]
        ]

        # Determine which securities have passed their closing date
        is_closed = close_dates_df.loc[sec_names] <= target.now

        # Close eligible positions
        for sec_name in is_closed[is_closed].index:
            target.close(sec_name, update=False)
            target.perm["closed"].add(sec_name)

        # Update the root after closing
        target.root.update(target.now)

        return True
