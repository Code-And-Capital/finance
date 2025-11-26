from bt.core.algo_base import Algo
import pandas as pd
import numpy as np


class WeighInvVol(Algo):
    """
    Algo that sets target weights using the inverse-volatility method.

    This technique is widely used in risk-parity and volatility-weighted
    portfolio construction. Assets with lower volatility (over the lookback
    window) receive higher weights, proportional to 1 / volatility.

    The algo looks back a specified period, computes returns of each selected
    asset, and calculates inverse-volatility weights using:
        `ffn.calc_inv_vol_weights()`

    Parameters
    ----------
    lookback : pd.DateOffset, optional
        Length of the lookback window for estimating volatility.
        Defaults to a 3-month lookback.
    lag : pd.DateOffset, optional
        Amount to lag the reference date (useful for avoiding lookahead bias
        when today's prices are unknown). Defaults to 0 days.

    Sets
    ----
    weights : dict or Series
        Stored in `target.temp["weights"]`.

    Requires
    --------
    selected : list[str]
        List of tickers selected by prior steps in the strategy.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ):
        """
        Initialize the inverse-volatility weighting algo.

        Parameters
        ----------
        lookback : pd.DateOffset
            Lookback period used for volatility estimation.
        lag : pd.DateOffset
            Lag applied to the current date before sampling price data.
        """
        super().__init__()
        self.lookback = lookback
        self.lag = lag

    def calc_inv_vol_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Compute portfolio weights proportional to the inverse of asset volatility.

        This method is commonly used in risk parity and inverse-volatility portfolios.
        Each asset's weight is proportional to 1 / std_dev(asset_returns), ensuring
        that less volatile assets receive higher weights.

        Assets with constant returns (zero volatility) or all NaNs are assigned NaN
        weights and excluded from the portfolio.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of asset returns, with columns representing assets and rows
            representing time periods.

        Returns
        -------
        pd.Series
            Series of weights indexed by asset name, summing to 1 for all valid
            assets. Assets with undefined volatility are assigned NaN.
        """
        # Compute standard deviation (volatility) per column
        std_vol = returns.std(ddof=1)

        # Compute inverse volatility
        inv_vol = 1.0 / std_vol

        # Replace infinite values (from zero volatility) with NaN
        inv_vol.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Normalize to sum to 1
        total_inv_vol = inv_vol.sum(skipna=True)
        weights = inv_vol / total_inv_vol

        return weights

    def __call__(self, target) -> bool:
        """
        Compute inverse-volatility weights for the currently selected assets.

        Parameters
        ----------
        target : StrategyBase
            Strategy context object, providing:
            - `target.now`: current timestamp
            - `target.temp["selected"]`: list of tickers selected
            - `target.universe`: historical pricing DataFrame

        Returns
        -------
        bool
            Always returns True after setting weights (even if empty).
        """
        selected = target.temp.get("selected", [])

        # No assets selected → no weights
        if not selected:
            target.temp["weights"] = {}
            return True

        # Only one asset → full weight
        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        # Calculate reference date with possible lag
        ref_date = target.now - self.lag

        # Extract price history for the lookback window
        price_window = target.universe.loc[
            ref_date - self.lookback : ref_date, selected
        ]

        # Convert to returns, drop NaNs, compute inverse-vol weights
        returns = price_window.pct_change().iloc[1:]

        if returns.empty:
            # Safeguard: if there's no valid return data, fall back to equal weights
            w = 1.0 / len(selected)
            target.temp["weights"] = {x: w for x in selected}
            return True

        inv_vol_weights = self.calc_inv_vol_weights(returns).dropna()

        # Save final weights
        target.temp["weights"] = inv_vol_weights

        return True
