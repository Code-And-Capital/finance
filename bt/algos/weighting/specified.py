from bt.core.algo_base import Algo
import pandas as pd


class WeighSpecified(Algo):
    """
    Algo that assigns portfolio weights based on a user-provided mapping.

    The algorithm sets ``temp['weights']`` on the target using a predefined
    dict of ticker→weight pairs. This allows explicit control over target
    allocations instead of calculating them dynamically.

    Sets:
        * weights

    Attributes:
        weights (dict): Mapping of ticker → desired weight.
    """

    def __init__(self, **weights: float):
        """
        Initialize the WeighSpecified algorithm.

        Parameters
        ----------
        **weights : float
            Arbitrary keyword arguments where each key is a ticker symbol
            and each value is its desired portfolio weight.
        """
        super().__init__()
        self.weights = weights

    def __call__(self, target) -> bool:
        """
        Apply the specified weights to the target.

        Parameters
        ----------
        target : StrategyBase
            The strategy instance whose temporary state will be modified.

        Returns
        -------
        bool
            Always returns True to indicate successful execution.
        """
        target.temp["weights"] = self.weights.copy()
        return True


class WeighTarget(Algo):
    """
    Algo that assigns portfolio target weights from a time-indexed DataFrame.

    This algorithm allows variable-frequency rebalancing based on when weights
    are available in the provided DataFrame:

    - If the weights DataFrame has the same frequency as the price data,
      the portfolio is rebalanced every period (e.g., daily).
    - If the weights DataFrame contains only month-end dates, rebalancing
      happens only at month-end.
    - On dates where no target weight data exists, the algo returns False,
      meaning the strategy will not rebalance.

    Parameters
    ----------
    weights : pd.DataFrame or str
        A DataFrame of target weights indexed by date and with tickers as
        columns, OR a string key which will be used to fetch the DataFrame
        via `target.get_data()`. Using a string is the recommended pattern.

    Sets
    ----
    weights : Series
        Stored in `target.temp["weights"]` on dates where target weights exist.

    Requires
    --------
    The strategy must provide weight data via:
        • A static DataFrame passed directly, or
        • `target.get_data(weights_name)` if `weights` is a string.
    """

    def __init__(self, weights: pd.DataFrame | str):
        """
        Initialize the WeighTarget algorithm.

        Parameters
        ----------
        weights : pd.DataFrame or str
            If a DataFrame is provided, it will be used directly.
            If a string is provided, it will be used as the data key for
            retrieving the DataFrame via `target.get_data()`.
        """
        super().__init__()

        if isinstance(weights, pd.DataFrame):
            self.weights_name = None
            self.weights = weights
        else:
            self.weights_name = weights
            self.weights = None

    def __call__(self, target) -> bool:
        """
        Assign target weights if weights exist for the current date.

        Parameters
        ----------
        target : StrategyBase
            The strategy instance containing context, including:
            • `target.now` - the current timestamp
            • `target.get_data()` - for retrieving weight frames

        Returns
        -------
        bool
            True if weights are assigned for the current date, otherwise False.
        """
        # retrieve the DataFrame depending on initialization mode
        if self.weights_name is None:
            weights = self.weights
        else:
            weights = target.get_data(self.weights_name)

        # no weights available → no rebalancing
        if weights is None or weights.empty:
            return False

        # check if weights exist for this timestamp
        if target.now in weights.index:
            row = weights.loc[target.now]

            # Drop NaNs to avoid invalid tickers and assign to temp
            target.temp["weights"] = row.dropna()

            return True

        # No weight entry for current date
        return False
