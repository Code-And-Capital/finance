from bt.core.algo_base import Algo
import pandas as pd
import numpy as np


class MovingAverage(Algo):
    """
    Abstract base class for cross-sectional moving-average estimators.

    This class handles common logic for lag-adjusted evaluation times,
    universe resolution, and data availability checks. Subclasses must implement
    the `_compute_average` method to define the weighting scheme.

    Parameters
    ----------
    lag : pandas.DateOffset, default pd.DateOffset(days=0)
        Time offset applied to `target.now` to prevent look-ahead bias. All
        calculations use data available up to `target.now - lag`.

    dict_name : str
        Key under which the computed moving-average vector is stored in
        `target.temp`.
    """

    def __init__(self, lag=pd.DateOffset(days=0), dict_name="moving_average"):
        super().__init__()
        self.lag = lag
        self.dict_name = dict_name

    def __call__(self, target):
        """
        Compute and store the moving average for the current target state.

        Parameters
        ----------
        target : object
            Execution context provided by the strategy engine. Expected to expose:
            - now : pandas.Timestamp
            - universe : pandas.DataFrame (index = datetime, columns = assets)
            - temp : dict-like container for intermediate results

        Returns
        -------
        bool
            True if the moving average was successfully computed and stored.
            False if insufficient historical data was available.
        """
        t0 = target.now - self.lag

        # Determine active universe
        if "selected" in target.temp:
            assets = list(target.temp["selected"])
        else:
            assets = list(target.universe.columns)

        prices = target.universe[assets]

        # Restrict to history available up to t0
        hist = prices.loc[:t0]

        if hist.empty:
            return False

        # Delegate to subclass for computing the average
        ma = self._compute_average(hist, t0)

        if ma is None or ma.empty:
            return False

        target.temp[self.dict_name] = ma
        return True

    def _compute_average(self, hist: pd.DataFrame, t0: pd.Timestamp) -> pd.Series:
        """
        Abstract method to compute the moving average for a given history.

        Parameters
        ----------
        hist : pd.DataFrame
            Historical prices up to evaluation time.
        t0 : pandas.Timestamp
            Evaluation timestamp.

        Returns
        -------
        pd.Series
            Cross-sectional moving-average levels indexed by asset.
        """
        raise NotImplementedError("Subclasses must implement `_compute_average`.")


# ================================
# SimpleMovingAverage subclass
# ================================


class SimpleMovingAverage(MovingAverage):
    """
    Cross-sectional trend-following signal based on a rolling mean or median.

    The signal computes a rolling price benchmark over a specified lookback
    window. Assets whose current price is above the benchmark are considered
    to be trending positively.

    Parameters
    ----------
    lookback : pandas.DateOffset, default pd.DateOffset(months=3)
        Length of the historical window used to compute the rolling average.

    measure : {"mean", "median"}, default "mean"
        Aggregation function used to compute the moving average. The median
        provides additional robustness to outliers.

    lag : pandas.DateOffset, default pd.DateOffset(days=0)
        Time offset applied to `target.now` to prevent look-ahead bias.

    dict_name : str, default "moving_average"
        Key under which the computed moving-average vector is stored in
        `target.temp`.
    """

    def __init__(
        self,
        lookback=pd.DateOffset(months=3),
        measure="mean",
        lag=pd.DateOffset(days=0),
        dict_name="moving_average",
    ):
        super().__init__(lag=lag, dict_name=dict_name)

        if measure not in {"mean", "median"}:
            raise ValueError(
                f"Unsupported measure '{measure}'. Use 'mean' or 'median'."
            )

        self.lookback = lookback
        self.measure = measure

    def _compute_average(self, hist: pd.DataFrame, t0: pd.Timestamp) -> pd.Series:
        window = hist.loc[t0 - self.lookback : t0]

        if window.empty:
            return pd.Series(dtype=float)

        if self.measure == "mean":
            return window.mean()
        else:
            return window.median()


# ================================
# ExponentialWeightedMovingAverage subclass
# ================================


class ExponentialWeightedMovingAverage(MovingAverage):
    """
    Cross-sectional exponential weighted moving-average estimator.

    This Algo computes an exponentially weighted moving average (EWMA) for each
    asset in the active universe using a half-life parameterization and stores
    the result in `target.temp`.

    The EWMA places exponentially decaying weights on historical prices, with
    more recent observations receiving greater weight. The decay rate is
    controlled via the half-life, which provides a direct and interpretable
    notion of the signal's effective memory.

    Parameters
    ----------
    half_life : int or float
        Half-life of the exponential decay, expressed in number of periods.
        The half-life defines how many periods it takes for the weight of an
        observation to decay to one half of its original value.

    lag : pandas.DateOffset, default pd.DateOffset(days=0)
        Time offset applied to `target.now` to prevent look-ahead bias.

    dict_name : str, default "ewma"
        Key under which the computed EWMA vector is stored in `target.temp`.
    """

    def __init__(
        self,
        half_life: float,
        lag=pd.DateOffset(days=0),
        dict_name="ewma",
    ):
        super().__init__(lag=lag, dict_name=dict_name)

        if half_life <= 0:
            raise ValueError("half_life must be strictly positive.")

        self.half_life = half_life
        # Convert half-life to smoothing parameter alpha
        self.alpha = 1.0 - 2.0 ** (-1.0 / self.half_life)

    def _compute_average(self, hist: pd.DataFrame, t0: pd.Timestamp) -> pd.Series:
        if hist.empty:
            return pd.Series(dtype=float)

        ewma = hist.ewm(alpha=self.alpha, adjust=False, min_periods=1).mean()
        # Select last available row for cross-section
        return ewma.iloc[-1]
