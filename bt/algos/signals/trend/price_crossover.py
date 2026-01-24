from bt.core.algo_base import Algo
import pandas as pd


class TrendSignalBase(Algo):
    """
    Base class for cross-sectional trend-following signals.

    This abstract class handles common operations for trend signals, including:
    - Determining the evaluation timestamp with a lag to prevent look-ahead bias
    - Selecting the active universe of assets
    - Retrieving the latest available prices
    - Storing the resulting cross-sectional boolean signal in `target.temp["selected"]`

    Subclasses must implement the `_compute_signal` method, which defines the
    logic for selecting assets based on their prices and/or benchmark indicators.
    """

    def __init__(self, lag=pd.DateOffset(days=0)):
        """
        Initialize the trend signal base.

        Parameters
        ----------
        lag : pandas.DateOffset, optional
            Time offset applied to `target.now` to avoid look-ahead bias. All
            calculations are performed using data available up to `target.now - lag`.
        """
        super().__init__()
        self.lag = lag

    def __call__(self, target):
        """
        Execute the trend signal computation for the current target state.

        This method handles the evaluation timestamp, universe selection,
        price retrieval, and storage of the resulting signal. The specific
        selection logic is delegated to the subclass via `_compute_signal`.

        Parameters
        ----------
        target : object
            Execution context provided by the backtesting or live-trading
            framework. Expected to provide:
            - `now` : pandas.Timestamp, current evaluation timestamp
            - `universe` : pandas.DataFrame, index = datetime, columns = assets
            - `temp` : dict-like storage for intermediate state

        Returns
        -------
        bool
            True if the signal was successfully computed and stored; False if
            insufficient data or assets are available.
        """
        t0 = target.now - self.lag

        # Determine active universe
        if "selected" in target.temp:
            assets = list(target.temp["selected"])
        else:
            assets = list(target.universe.columns)

        if not assets:
            return False

        prices = target.universe[assets]

        # Check that price history exists
        if prices.index[0] > t0:
            return False

        # Latest available prices
        latest_prices = prices.loc[:t0].iloc[-1]

        # Delegate to subclass for signal computation
        trend = self._compute_signal(target, latest_prices, assets)

        # Store cross-sectional selection as dictionary
        target.temp["selected"] = trend.to_dict()

        return True

    def _compute_signal(self, target, latest_prices, assets):
        """
        Subclasses must override this method to implement specific selection logic.

        Parameters
        ----------
        target : object
            Backtesting context with universe, temp storage, etc.
        latest_prices : pd.Series
            Latest available prices for the selected assets.
        assets : list
            Names of the assets to evaluate.

        Returns
        -------
        pd.Series
            Boolean Series indicating which assets are selected (True = selected).
        """
        raise NotImplementedError("Subclasses must implement this method")


class PriceCrossOverSignal(TrendSignalBase):
    """
    Cross-sectional price-over-benchmark signal.

    Generates a boolean signal by comparing the latest price of each asset
    to a reference benchmark (e.g., SMA, EWMA). Assets with prices above
    the benchmark are considered trending positively and selected.

    The resulting signal is stored in `target.temp["selected"]` for downstream use.
    """

    def __init__(self, ma_name="moving_average", lag=pd.DateOffset(days=0)):
        """
        Initialize the price crossover signal.

        Parameters
        ----------
        ma_name : str, optional
            Key in `target.temp` containing the reference benchmark series.
        lag : pandas.DateOffset, optional
            Offset applied to evaluation time to prevent look-ahead bias.
        """
        super().__init__(lag=lag)
        self.ma_name = ma_name

    def _compute_signal(self, target, latest_prices, assets):
        """
        Compute the price-over-benchmark signal.

        Parameters
        ----------
        target : object
            Backtesting context containing benchmark data in `target.temp`.
        latest_prices : pd.Series
            Latest available prices of the selected assets.
        assets : list
            Names of the assets to evaluate.

        Returns
        -------
        pd.Series
            Boolean Series indicating which assets are above the benchmark.
        """
        if self.ma_name not in target.temp:
            raise ValueError(
                f"Reference benchmark '{self.ma_name}' not found in target.temp"
            )

        ref = target.temp[self.ma_name].loc[assets]
        trend = latest_prices > ref
        return trend[trend]


class DualMACrossoverSignal(TrendSignalBase):
    """
    Dual moving average crossover signal.

    Generates a boolean signal by comparing a short-term moving average
    to a long-term moving average. Assets where the short-term average
    exceeds the long-term average are considered trending positively.

    The resulting signal is stored in `target.temp["selected"]` for downstream use.
    """

    def __init__(
        self, short_name="ma_short", long_name="ma_long", lag=pd.DateOffset(days=0)
    ):
        """
        Initialize the dual moving average crossover signal.

        Parameters
        ----------
        short_name : str, optional
            Key in `target.temp` containing the short-term moving average.
        long_name : str, optional
            Key in `target.temp` containing the long-term moving average.
        lag : pandas.DateOffset, optional
            Offset applied to evaluation time to prevent look-ahead bias.
        """
        super().__init__(lag=lag)
        self.short_name = short_name
        self.long_name = long_name

    def _compute_signal(self, target, latest_prices, assets):
        """
        Compute the dual moving average crossover signal.

        Parameters
        ----------
        target : object
            Backtesting context containing the moving averages in `target.temp`.
        latest_prices : pd.Series
            Latest available prices of the selected assets (not used in this signal).
        assets : list
            Names of the assets to evaluate.

        Returns
        -------
        pd.Series
            Boolean Series indicating which assets have a bullish crossover
            (short-term MA > long-term MA).
        """
        for name in [self.short_name, self.long_name]:
            if name not in target.temp:
                raise ValueError(f"Benchmark '{name}' not found in target.temp")

        short_ma = target.temp[self.short_name].loc[assets]
        long_ma = target.temp[self.long_name].loc[assets]

        trend = short_ma > long_ma
        return trend[trend]
