from bt.core.algo_base import Algo
import pandas as pd
import numpy as np
import sklearn


class TargetVol(Algo):
    """
    Adjust portfolio weights to achieve a target annualized volatility.

    This Algo rescales `temp['weights']` so that the portfolio's
    realized volatility matches the desired `target_volatility`.
    Useful for risk-parity-like scaling or volatility targeting strategies.

    Parameters
    ----------
    target_volatility : float or dict
        Annualized volatility to target. If a dict is provided, each asset can have a separate target.
    lookback : pd.DateOffset, optional
        Lookback period to estimate covariance. Default 3 months.
    lag : pd.DateOffset, optional
        Time lag before calculating covariance to avoid lookahead bias. Default 0 days.
    covar_method : str, optional
        Method to estimate covariance matrix: "standard" or "ledoit-wolf". Default "standard".
    annualization_factor : int, optional
        Factor to annualize volatility (e.g., 252 for daily data). Default 252.

    Updates
    -------
    temp['weights'] : dict
        Rescaled weights to match the target volatility.

    Requires
    --------
    temp['weights'] : dict
        Initial weights must be set before calling this Algo.
    """

    def __init__(
        self,
        target_volatility: float | dict,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
        covar_method: str = "standard",
        annualization_factor: int = 252,
    ):
        super().__init__()
        self.target_volatility = target_volatility
        self.lookback = lookback
        self.lag = lag
        self.covar_method = covar_method
        self.annualization_factor = annualization_factor

    def __call__(self, target) -> bool:
        """
        Rescale current weights to achieve the target volatility.

        Parameters
        ----------
        target : StrategyBase
            The strategy context. Must have `temp['weights']` and `universe`.

        Returns
        -------
        bool
            Always True after adjusting weights.
        """
        current_weights = target.temp.get("weights", {})
        if not current_weights:
            # Nothing to scale
            return True

        selected = list(current_weights.keys())
        ref_date = target.now - self.lag
        price_window = target.universe.loc[
            ref_date - self.lookback : ref_date, selected
        ]
        returns = price_window.pct_change().iloc[1:]

        # Covariance estimation
        if self.covar_method == "ledoit-wolf":
            covar = sklearn.covariance.ledoit_wolf(returns)[0]
        elif self.covar_method == "standard":
            covar = returns.cov().values
        else:
            raise NotImplementedError(
                f"covar_method '{self.covar_method}' not implemented."
            )

        # Align weights to covariance matrix columns
        weights = pd.Series([current_weights[k] for k in selected], index=selected)

        # Current portfolio volatility
        port_var = np.dot(weights.values.T, np.dot(covar, weights.values))
        port_vol = np.sqrt(port_var * self.annualization_factor)

        # Convert scalar target_volatility to dict if necessary
        if isinstance(self.target_volatility, (float, int)):
            target_vol_dict = {k: self.target_volatility for k in selected}
        else:
            target_vol_dict = self.target_volatility

        # Rescale weights to achieve target volatility per asset
        for k in selected:
            if k in target_vol_dict:
                weights[k] = weights[k] * target_vol_dict[k] / port_vol

        target.temp["weights"].update(weights.to_dict())
        return True
