import pandas as pd
from typing import List, Union, Any
from bt.core import Algo


class RunOnDate(Algo):
    """
    Algo that triggers only on a specific set of dates.

    This is useful for backtests or strategies where certain actions
    (e.g., dividends, manual rebalancing, reporting) should occur on
    predefined dates.

    Args:
        dates : list of str, datetime, or pandas.Timestamp
            The set of dates on which the algo should run. Each date
            will be converted to a pandas.Timestamp internally.

    Examples
    --------
    Run on two specific dates:
        RunOnDate('2025-01-01', '2025-06-30')

    Run on multiple datetime objects:
        RunOnDate(datetime(2025,1,1), datetime(2025,6,30))
    """

    def __init__(self, *dates: Union[str, pd.Timestamp, Any]):
        """
        Initialize RunOnDate with a list of dates.

        Parameters
        ----------
        *dates : variable length argument list
            Dates to trigger the algo. Can be strings (YYYY-MM-DD),
            pandas.Timestamp, or datetime objects. They will be parsed
            into pandas.Timestamp internally.
        """
        super().__init__()
        self.dates: List[pd.Timestamp] = [pd.to_datetime(d) for d in dates]

    def __call__(self, target: Any) -> bool:
        """
        Execute the algo and determine if the current date is in the
        set of target dates.

        Parameters
        ----------
        target : bt.backtest.Target
            The backtest target containing the `now` attribute (current date).

        Returns
        -------
        bool
            True if `target.now` is in the specified dates, False otherwise.
        """
        return target.now in self.dates
