from bt.core.algo_base import Algo
import pandas as pd
import numpy as np
import sklearn
from scipy.optimize import minimize


class WeighMeanVar(Algo):
    """
    Assign portfolio weights using mean-variance (Markowitz) optimization.

    This Algo sets `temp['weights']` by maximizing the Sharpe ratio of
    the portfolio given expected returns, covariance, and optional bounds.
    It is a Python implementation of Markowitz's Modern Portfolio Theory.

    See:
        http://en.wikipedia.org/wiki/Modern_portfolio_theory#The_efficient_frontier_with_no_risk-free_asset

    Parameters
    ----------
    lookback : pd.DateOffset, optional
        Lookback period for estimating returns and volatility. Default 3 months.
    bounds : tuple(float, float), optional
        Minimum and maximum allowable weights for each asset. Default (0.0, 1.0).
    covar_method : str, optional
        Covariance estimation method. Options:
        - "ledoit-wolf" (default)
        - "standard"
    rf : float, optional
        Risk-free rate used in Sharpe ratio calculation. Default 0.0.
    lag : pd.DateOffset, optional
        Lag applied to the current date to avoid lookahead bias. Default 0 days.

    Sets
    ----
    weights : dict or pd.Series
        Stored in `target.temp["weights"]`.

    Requires
    --------
    selected : list[str]
        List of selected tickers to optimize.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        bounds: tuple[float, float] = (0.0, 1.0),
        covar_method: str = "ledoit-wolf",
        rf: float = 0.0,
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ):
        super().__init__()
        self.lookback = lookback
        self.bounds = bounds
        self.covar_method = covar_method
        self.rf = rf
        self.lag = lag

    def calc_mean_var_weights(
        self,
        returns: pd.DataFrame,
        weight_bounds: tuple[float, float] = (0.0, 1.0),
        rf: float = 0.0,
        covar_method: str = "ledoit-wolf",
        options: dict | None = None,
    ) -> pd.Series:
        """
        Calculate mean-variance optimized portfolio weights using Sharpe ratio maximization.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns for selected assets.
        weight_bounds : tuple(float, float), optional
            Minimum and maximum weights for each asset. Default (0.0, 1.0).
        rf : float, optional
            Risk-free rate used in utility calculation. Default 0.0.
        covar_method : str, optional
            Covariance estimation method. Default 'ledoit-wolf'.
        options : dict, optional
            Options for optimizer (e.g., {'maxiter': 10000}).

        Returns
        -------
        pd.Series
            Optimized weights indexed by asset name.

        Raises
        ------
        Exception
            If the optimization fails.
        NotImplementedError
            If the covariance method is not supported.
        """

        def fitness(
            weights: np.ndarray, exp_rets: pd.Series, covar: np.ndarray, rf: float
        ) -> float:
            mean = np.dot(exp_rets.values, weights)
            var = np.dot(weights, covar.dot(weights))
            # Maximize Sharpe ratio (negate because optimizer minimizes)
            return -(mean - rf) / np.sqrt(var)

        n_assets = len(returns.columns)
        exp_rets = returns.mean()

        # Covariance matrix
        if covar_method == "ledoit-wolf":
            covar = sklearn.covariance.ledoit_wolf(returns)[0]
        elif covar_method == "standard":
            covar = returns.cov().values
        else:
            raise NotImplementedError(f"covar_method '{covar_method}' not implemented.")

        # Initial weights and bounds
        weights0 = np.ones(n_assets) / n_assets
        bounds = [weight_bounds] * n_assets
        constraints = {"type": "eq", "fun": lambda W: np.sum(W) - 1.0}

        result = minimize(
            fitness,
            weights0,
            args=(exp_rets, covar, rf),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        if not result.success:
            raise Exception(result.message)

        return pd.Series(result.x, index=returns.columns)

    def __call__(self, target) -> bool:
        """
        Compute mean-variance optimized weights for the selected assets and
        store them in `target.temp['weights']`.

        Parameters
        ----------
        target : StrategyBase
            Strategy context with `temp["selected"]` and `universe` attributes.

        Returns
        -------
        bool
            Always True after setting weights.
        """
        selected = target.temp.get("selected", [])

        if not selected:
            target.temp["weights"] = {}
            return True

        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        ref_date = target.now - self.lag
        price_window = target.universe.loc[
            ref_date - self.lookback : ref_date, selected
        ]
        returns = price_window.pct_change().iloc[1:]

        weights = self.calc_mean_var_weights(
            returns,
            weight_bounds=self.bounds,
            rf=self.rf,
            covar_method=self.covar_method,
        )

        target.temp["weights"] = weights.dropna()
        return True
