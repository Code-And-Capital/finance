from bt.core import Algo
import pandas as pd


class StatTotalReturn(Algo):
    """
    Computes total return for each selected security over a specified period.

    This Algo sets ``temp['stat']`` to the total return of each ticker in
    ``temp['selected']``. The total return is computed over the interval:

        [ now - lookback - lag  →  now - lag ]

    using ffn's ``calc_total_return`` method.

    Example:
        If ``lookback = 3 months`` and ``lag = 2 days``, and today is T:
            - End date (t0) = T - 2 days
            - Start date     = t0 - 3 months

    Requirements:
        * ``temp['selected']`` must already exist (usually set by SelectAll or similar).

    Side Effects:
        * Sets ``temp['stat']`` to a pandas Series indexed by ticker.

    Args:
        lookback (pd.DateOffset):
            Lookback window used for total return computation.
        lag (pd.DateOffset):
            Optional lag applied to the evaluation date.

    Returns:
        bool: True if successful, False if insufficient data exists.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ):
        super().__init__()
        self.lookback = lookback
        self.lag = lag

    def calc_total_return(self, prices):
        """
        Computes total return from a price Series or DataFrame.

        Total return is defined as:

            (last_price / first_price) - 1

        Expected input:
            A pandas Series (single asset) or DataFrame (multiple assets) indexed
            in chronological order.

        Behavior:
            - Automatically handles missing data by forward-filling internally.
            - If the first or last value is missing after forward-fill, return NaN.
            - For DataFrames, returns a Series of total returns per column.

        Args:
            prices (pd.Series or pd.DataFrame):
                Price history to compute total return from.

        Returns:
            pd.Series or float:
                Total return(s) over the period.
        """
        # Forward fill to handle internal gaps
        prc = prices.ffill()

        # Determine start and end values
        first = prc.iloc[0]
        last = prc.iloc[-1]

        # Replace zero first values only for DataFrame columns
        if hasattr(first, "index"):  # DataFrame
            first = first.replace(0, pd.NA)
        elif first == 0:  # Series
            return float("nan")

        return (last / first) - 1

    def __call__(self, target) -> bool:
        selected = target.temp["selected"]

        # Determine end of evaluation window
        t0 = target.now - self.lag

        # Not enough historical data → fail and skip this Algo
        first_index = target.universe[selected].index[0]
        if first_index > t0:
            return False

        # Slice data: [t0 - lookback, t0]
        prc = target.universe.loc[t0 - self.lookback : t0, selected]

        # Compute total return for each column (ticker)
        target.temp["stat"] = self.calc_total_return(prc)

        return True
