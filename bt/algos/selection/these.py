from typing import Iterable, Optional, Union, List
from bt.core import Algo
import pandas as pd


class SelectThese(Algo):
    """
    Selection Algo that sets ``temp['selected']`` to a specified list of tickers.

    This Algo selects only the provided tickers and stores them in
    ``temp['selected']``. By default, it filters out securities with missing,
    zero, or negative prices on the current date unless overridden.

    Args:
        tickers (Iterable[str]): Iterable of tickers to select.
        include_no_data (bool, optional): If True, include all provided
            tickers regardless of missing price data. Defaults to False.
        include_negative (bool, optional): If True, include securities with
            zero or negative prices. Only applies when ``include_no_data`` is False.
            Defaults to False.

    Sets:
        temp['selected'] (list[str]): The list of selected tickers.

    Example:
        >>> algo = SelectThese(["AAPL", "MSFT"], include_no_data=False)
        >>> algo(target)
        >>> target.temp["selected"]
    """

    def __init__(
        self,
        tickers: Iterable[str],
        include_no_data: bool = False,
        include_negative: bool = False,
    ) -> None:
        """
        Initialize the SelectThese Algo.

        Args:
            tickers (Iterable[str]): Tickers to select.
            include_no_data (bool): Include tickers with no current data.
            include_negative (bool): Include tickers with zero or negative prices.
        """
        super().__init__()
        self.tickers = list(tickers)
        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target) -> bool:
        """
        Select the specified tickers and store them in ``temp['selected']``.

        Args:
            target (Any): Strategy/backtest object containing ``universe``,
                ``now``, and ``temp``.

        Returns:
            bool: Always returns True.
        """
        # Filter tickers that exist in the universe
        available_tickers = [t for t in self.tickers if t in target.universe.columns]

        if self.include_no_data:
            target.temp["selected"] = available_tickers
            return True

        # Extract current-row prices for the tickers
        row = target.universe.loc[target.now, available_tickers].dropna()

        if self.include_negative:
            target.temp["selected"] = list(row.index)
        else:
            target.temp["selected"] = list(row[row > 0].index)

        return True


class SelectWhere(Algo):
    """
    Selects securities based on a boolean signal DataFrame.

    Selects tickers where the signal is True on the current date (target.now).
    The signal may be provided directly as a DataFrame or as a dataset name
    to be retrieved using target.get_data(name).

    Args:
        signal (str | DataFrame): Boolean selection signal.
        include_no_data (bool): If False, excludes tickers with no data at now.
        include_negative (bool): If False, excludes tickers with <= 0 prices.

    Sets:
        selected
    """

    def __init__(
        self,
        signal: Union[str, pd.DataFrame],
        include_no_data: bool = False,
        include_negative: bool = False,
    ) -> None:
        """
        Initialize the SelectWhere algorithm.

        Parameters:
            signal (str | DataFrame): Boolean selection signal.
            include_no_data (bool): Exclude missing data if False.
            include_negative (bool): Exclude <= 0 prices if False.
        """
        super().__init__()

        if isinstance(signal, pd.DataFrame):
            self.signal = signal
            self.signal_name = None
        else:
            self.signal = None
            self.signal_name = signal

        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target) -> bool:
        """
        Execute the selection logic.

        Steps:
            1. Resolve signal (DF or retrieved via name).
            2. Use the row at target.now.
            3. Select tickers with True signal.
            4. Apply price-based filtering.
            5. Store in temp["selected"].

        Parameters:
            target: Strategy/backtest object providing universe, now, temp, get_data.

        Returns:
            True (algo always passes control to next).
        """
        if self.signal_name is None:
            signal_df = self.signal
        else:
            signal_df = target.get_data(self.signal_name)

        if target.now not in signal_df.index:
            return True

        row = signal_df.loc[target.now]
        selected = list(row[row == True].index)

        if not self.include_no_data:
            universe = target.universe.loc[target.now, selected].dropna()

            if self.include_negative:
                selected = list(universe.index)
            else:
                selected = list(universe[universe > 0].index)

        target.temp["selected"] = selected
        return True
