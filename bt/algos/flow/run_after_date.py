import pandas as pd
from typing import Any, Union
from bt.core.algo_base import Algo


class RunAfterDate(Algo):
    """
    Algo that triggers only after a specific date has passed.

    This is useful for strategies that require an initial warm-up period,
    such as calculating trailing averages or waiting until sufficient data
    has accumulated before taking action.

    Args:
        date : str, datetime, or pandas.Timestamp
            The date after which the algo should start running.

    Examples
    --------
    Start trading after January 1st, 2025:
        RunAfterDate('2025-01-01')

    Using a datetime object:
        RunAfterDate(datetime(2025, 1, 1))
    """

    def __init__(self, date: Union[str, pd.Timestamp, Any]):
        """
        Initialize RunAfterDate with the specified start date.

        Parameters
        ----------
        date : str, datetime, or pandas.Timestamp
            The date after which the algo should start executing.
            It will be parsed internally to a pandas.Timestamp.
        """
        super().__init__()
        self.date: pd.Timestamp = pd.to_datetime(date)

    def __call__(self, target: Any) -> bool:
        """
        Execute the algo and determine if the current date is past
        the specified start date.

        Parameters
        ----------
        target : bt.backtest.Target
            The backtest target containing the `now` attribute (current date).

        Returns
        -------
        bool
            True if `target.now` is after the specified date, False otherwise.
        """
        return target.now > self.date
