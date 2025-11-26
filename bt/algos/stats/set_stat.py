from bt.core import Algo
import pandas as pd
from typing import Union


class SetStat(Algo):
    """
    Sets temp['stat'] for downstream algorithms (e.g., SelectN).

    This Algo either uses a precomputed DataFrame passed directly, or
    fetches a dataset by name via target.get_data. The statistic used
    is optionally lagged by the provided DateOffset.

    Args:
        stat (str | pd.DataFrame): The source of the statistic.
            - If str, fetched via target.get_data(stat)
            - If DataFrame, used directly
        lag (pd.DateOffset): The lag interval. temp['stat'] for today
            is taken from today - lag.

    Sets:
        temp['stat']
    """

    def __init__(
        self, stat: Union[str, pd.DataFrame], lag: pd.DateOffset = pd.DateOffset(days=0)
    ) -> None:
        """
        Initialize the SetStat algorithm.

        Parameters:
            stat (str | pd.DataFrame): Statistic source (name or DataFrame)
            lag (pd.DateOffset): Lag to apply when setting temp['stat']
        """
        super().__init__()
        self.lag = lag

        if isinstance(stat, pd.DataFrame):
            self.stat = stat
            self.stat_name = None
        else:
            self.stat = None
            self.stat_name = stat

    def __call__(self, target) -> bool:
        """
        Apply the statistic to temp['stat'].

        Steps:
        1. Fetch the DataFrame (direct or via target.get_data)
        2. Apply lag to determine the date for selection
        3. Store the row for the date in temp['stat']

        Parameters:
            target: Strategy/backtest container with 'now', 'temp', and 'get_data'

        Returns:
            bool: True if successfully set, False if lagged date is not available
        """
        stat_df = (
            self.stat if self.stat_name is None else target.get_data(self.stat_name)
        )

        t0 = target.now - self.lag
        if t0 not in stat_df.index:
            return False

        target.temp["stat"] = stat_df.loc[t0]
        return True
