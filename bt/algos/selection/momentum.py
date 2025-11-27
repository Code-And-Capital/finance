from bt.core.algo_base import AlgoStack
from bt.algos.selection.rank import SelectN
from bt.algos.stats.returns import StatTotalReturn
import pandas as pd


class SelectMomentum(AlgoStack):
    """
    Selects the top N tickers based on total return over a specified lookback period.

    This class is a thin wrapper around an AlgoStack consisting of:
        1. StatTotalReturn  - computes total return over a lookback/lag period
        2. SelectN          - selects the top N based on that total return

    Important:
        A selector such as SelectAll() or SelectThese() should be run before
        SelectMomentum(), because StatTotalReturn reads from temp['selected'].

    Args:
        n (int):
            Number of items to select.
        lookback (pd.DateOffset):
            Lookback period used to compute the total return.
        lag (pd.DateOffset):
            Optional lag shift applied to the lookback period.
        sort_descending (bool):
            If True, highest momentum is selected first.
        all_or_none (bool):
            If True, select only if N items are available. Otherwise return [].

    Sets:
        temp['selected']

    Requires:
        temp['selected'] before running StatTotalReturn
    """

    def __init__(
        self,
        n: int,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
        sort_descending: bool = True,
        all_or_none: bool = False,
    ):
        super().__init__(
            StatTotalReturn(lookback=lookback, lag=lag),
            SelectN(
                n=n,
                sort_descending=sort_descending,
                all_or_none=all_or_none,
                filter_selected=True,
            ),
        )
