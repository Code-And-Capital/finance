from bt.core import Algo
import pandas as pd


class SelectAll(Algo):
    """
    Selection Algo that stores all valid securities in ``temp['selected']``.

    This Algo selects all securities from the universe and saves them under
    ``temp['selected']`` for use by downstream Algos. By default, only
    securities with valid (non-NaN), positive prices on the current date
    are included.

    The behavior can be relaxed by enabling the arguments below.

    Args:
        include_no_data (bool, optional): If True, include all securities
            regardless of whether they have data on the current date.
            Defaults to False.

        include_negative (bool, optional): If True, include securities with
            zero or negative prices. Only used if ``include_no_data`` is False.
            Defaults to False.

    Sets:
        temp['selected'] (list[str]): The list of selected tickers.

    Example:
        >>> algo = SelectAll(include_no_data=False, include_negative=False)
        >>> algo(target)
        >>> target.temp['selected']
    """

    def __init__(
        self, include_no_data: bool = False, include_negative: bool = False
    ) -> None:
        """
        Initialize the SelectAll Algo.

        Args:
            include_no_data (bool): Whether to include securities with no data.
            include_negative (bool): Whether to include securities with
                zero or negative prices.
        """
        super().__init__()
        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target) -> bool:
        """
        Select all valid securities from the universe and store them in temp.

        Args:
            target (Any): The strategy or backtest object containing
                ``universe``, ``now``, and ``temp``.

        Returns:
            bool: Always returns True.
        """
        if self.include_no_data:
            target.temp["selected"] = target.universe.columns
        else:
            universe = target.universe.loc[target.now].dropna()

            if self.include_negative:
                target.temp["selected"] = list(universe.index)
            else:
                target.temp["selected"] = list(universe[universe > 0].index)

        return True
