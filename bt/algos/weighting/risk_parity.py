from bt.core.algo_base import Algo
import pandas as pd
import numpy as np
import pandas as pd
import sklearn.covariance


class WeighERC(Algo):
    """
    Assign portfolio weights using the Equal Risk Contribution (ERC) method.

    ERC is an extension of inverse volatility risk parity that incorporates
    correlations between asset returns. The resulting portfolio allocates
    weights so that each asset contributes equally to total portfolio risk,
    subject to diversification constraints.

    This approach produces a portfolio with volatility between that of the
    minimum variance and equally-weighted portfolios (Maillard 2008).

    See:
        https://en.wikipedia.org/wiki/Risk_parity

    Parameters
    ----------
    lookback : pd.DateOffset, optional
        Lookback period for estimating covariance. Default is 3 months.
    initial_weights : list[float], optional
        Starting asset weights for the iterative solution. Default is inverse vol.
    risk_weights : list[float], optional
        Target risk contribution weights. Default is equal weights.
    covar_method : str, optional
        Covariance estimation method. Default 'ledoit-wolf'.
    maximum_iterations : int, optional
        Maximum iterations in iterative solution. Default 100.
    tolerance : float, optional
        Tolerance level for convergence in iterative solution. Default 1e-8.
    lag : pd.DateOffset, optional
        Optional lag applied to the current date (avoids lookahead bias). Default 0 days.

    Sets
    ----
    weights : dict or pd.Series
        Stored in `target.temp["weights"]`.

    Requires
    --------
    selected : list[str]
        List of tickers previously selected by the strategy.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        initial_weights: list[float] | None = None,
        risk_weights: list[float] | None = None,
        covar_method: str = "ledoit-wolf",
        maximum_iterations: int = 100,
        tolerance: float = 1e-8,
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ):
        """
        Initialize the ERC weighting algorithm.
        """
        super().__init__()
        self.lookback = lookback
        self.initial_weights = initial_weights
        self.risk_weights = risk_weights
        self.covar_method = covar_method
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.lag = lag

    def _erc_weights_ccd(
        self,
        x0: np.ndarray,
        cov: np.ndarray,
        b: np.ndarray,
        maximum_iterations: int,
        tolerance: float,
    ) -> np.ndarray:
        """
        Solve the Equal Risk Contribution (ERC) portfolio using Cyclical Coordinate Descent (CCD).

        This algorithm finds portfolio weights `x` such that each asset contributes
        equally to total portfolio risk, given a covariance matrix and target risk
        contributions.

        Reference:
            Griveau-Billion, Theophile; Richard, Jean-Charles; Roncalli, Thierry (2013)
            "A Fast Algorithm for Computing High-Dimensional Risk Parity Portfolios"
            Available at SSRN: https://ssrn.com/abstract=2325255

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for asset weights (length n).
        cov : np.ndarray
            Covariance matrix of asset returns (n x n).
        b : np.ndarray
            Target risk contribution weights (length n).
        maximum_iterations : int
            Maximum number of iterations for convergence.
        tolerance : float
            Convergence tolerance. Algorithm stops if relative change in weights is below this value.

        Returns
        -------
        np.ndarray
            ERC weights normalized to sum to 1.

        Raises
        ------
        ValueError
            If no solution is found within the maximum number of iterations.
        """
        n_assets = len(x0)
        x = x0.copy()
        var = np.diagonal(cov)
        ctr = cov.dot(x)
        sigma_x = np.sqrt(x.T.dot(ctr))

        for iteration in range(maximum_iterations):
            for i in range(n_assets):
                alpha = var[i]
                beta = ctr[i] - x[i] * alpha
                gamma = -b[i] * sigma_x

                # Solve quadratic for updated weight
                x_tilde = (-beta + np.sqrt(beta**2 - 4 * alpha * gamma)) / (2 * alpha)
                x_i = x[i]

                # Update contributions and portfolio volatility incrementally
                ctr = ctr - cov[i] * x_i + cov[i] * x_tilde
                sigma_x = sigma_x**2 - 2 * x_i * cov[i].dot(x) + x_i**2 * var[i]
                x[i] = x_tilde
                sigma_x = np.sqrt(
                    sigma_x + 2 * x_tilde * cov[i].dot(x) - x_tilde**2 * var[i]
                )

            # Check convergence: relative squared change in weights
            if np.sum(((x - x0) / x.sum()) ** 2) < tolerance:
                return x / x.sum()

            x0 = x.copy()

        # Failed to converge
        raise ValueError(f"No solution found after {maximum_iterations} iterations.")

    def calc_erc_weights(
        self,
        returns: pd.DataFrame,
        initial_weights: np.ndarray | None = None,
        risk_weights: np.ndarray | None = None,
        covar_method: str = "ledoit-wolf",
        maximum_iterations: int = 100,
        tolerance: float = 1e-8,
    ) -> pd.Series:
        """
        Calculate Equal Risk Contribution (ERC) / risk parity weights for a set of assets.

        ERC weights aim to allocate portfolio risk equally across all assets,
        considering both volatility and correlation.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of asset returns, with columns as asset tickers.
        initial_weights : np.ndarray, optional
            Starting weights for the iterative solution. Default is inverse volatility.
        risk_weights : np.ndarray, optional
            Target risk contributions for each asset. Default is equal weights.
        covar_method : str, optional
            Covariance estimation method. Options:
            - "ledoit-wolf" (default): shrinkage estimator
            - "standard": sample covariance
        risk_parity_method : str, optional
            Risk parity solver. Options:
            - "ccd" (default): cyclical coordinate descent
            - "slsqp": SciPy's Sequential Least Squares Programming
        maximum_iterations : int, optional
            Maximum iterations for iterative solvers. Default 100.
        tolerance : float, optional
            Convergence tolerance for iterative solvers. Default 1e-8.

        Returns
        -------
        pd.Series
            ERC weights indexed by asset names.

        Raises
        ------
        NotImplementedError
            If `covar_method` or `risk_parity_method` is not supported.
        """
        n_assets = len(returns.columns)

        # Compute covariance matrix
        if covar_method == "ledoit-wolf":
            covar = sklearn.covariance.ledoit_wolf(returns)[0]
        elif covar_method == "standard":
            covar = returns.cov().values
        else:
            raise NotImplementedError(f"covar_method '{covar_method}' not implemented.")

        # Default initial weights to inverse volatility
        if initial_weights is None:
            inv_vol = 1.0 / np.sqrt(np.diagonal(covar))
            initial_weights = inv_vol / inv_vol.sum()

        # Default target risk contributions to equal
        if risk_weights is None:
            risk_weights = np.ones(n_assets) / n_assets

        # Solve for ERC weights
        erc_weights = self._erc_weights_ccd(
            initial_weights, covar, risk_weights, maximum_iterations, tolerance
        )

        return pd.Series(erc_weights, index=returns.columns, name="erc")

    def __call__(self, target) -> bool:
        """
        Compute ERC weights for the currently selected assets.

        Parameters
        ----------
        target : StrategyBase
            Strategy context with attributes:
            - `target.temp["selected"]`: list of tickers
            - `target.universe`: price history DataFrame

        Returns
        -------
        bool
            True after setting ERC weights in `target.temp["weights"]`.
        """
        selected = target.temp.get("selected", [])

        # No assets selected
        if not selected:
            target.temp["weights"] = {}
            return True

        # Single asset → full weight
        if len(selected) == 1:
            target.temp["weights"] = {selected[0]: 1.0}
            return True

        # Reference date with lag applied
        ref_date = target.now - self.lag

        # Extract relevant price history
        price_window = target.universe.loc[
            ref_date - self.lookback : ref_date, selected
        ]

        returns = price_window.pct_change().iloc[1:]

        # Compute ERC weights using ffn utility
        erc_weights = self.calc_erc_weights(
            returns,
            initial_weights=self.initial_weights,
            risk_weights=self.risk_weights,
            covar_method=self.covar_method,
            maximum_iterations=self.maximum_iterations,
            tolerance=self.tolerance,
        ).dropna()

        target.temp["weights"] = erc_weights

        return True
