from bt.core.algo_base import Algo
import numpy as np
import pandas as pd
from typing import Union


class LimitDeltas(Algo):
    """
    Restrict the change in portfolio weights from one period to the next.

    This Algo modifies `temp['weights']` so that each asset's weight change
    (delta) does not exceed a specified limit. Useful for controlling trading
    aggressiveness and reducing market impact.

    For example, if a security currently has weight 1.0 and the new target
    weight is 0.0, with a global limit of 0.1, the new weight will be
    adjusted to 0.9 instead of 0.0.

    Parameters
    ----------
    limit : float or dict, optional
        Maximum allowed weight change per asset. If float, applies globally.
        If dict, specifies per-asset limits.

    Updates
    -------
    temp['weights'] : dict
        Adjusted target weights respecting delta limits.

    Requires
    --------
    temp['weights'] : dict
        Initial target weights before applying delta limits.
    """

    def __init__(self, limit: float | dict = 0.1):
        super().__init__()
        self.limit = limit
        self.global_limit = not isinstance(limit, dict)

    def __call__(self, target) -> bool:
        """
        Apply delta limits to target weights.

        Parameters
        ----------
        target : StrategyBase
            Strategy context. Must have `temp['weights']` and `children` attributes.

        Returns
        -------
        bool
            Always True after applying limits.
        """
        tw = target.temp.get("weights", {})
        all_keys = set(target.children.keys()).union(tw.keys())

        for k in all_keys:
            current_weight = target.children[k].weight if k in target.children else 0.0
            target_weight = tw.get(k, 0.0)
            delta = target_weight - current_weight

            if self.global_limit:
                if abs(delta) > self.limit:
                    tw[k] = current_weight + self.limit * np.sign(delta)
            else:
                lmt = self.limit.get(k, None)
                if lmt is not None and abs(delta) > lmt:
                    tw[k] = current_weight + lmt * np.sign(delta)

        # Normalize weights to sum to 1
        total_weight = sum(tw.values())
        if total_weight > 0:
            tw = {k: w / total_weight for k, w in tw.items()}

        target.temp["weights"].update(tw)
        return True


class LimitWeights(Algo):
    """
    Restrict the maximum weight of any single asset in the portfolio.

    This Algo wraps limit_weights logic. It caps any asset weight at a
    specified `limit` and redistributes the excess proportionally among the
    other assets. Useful to avoid extreme allocations from upstream
    weighting algorithms.

    Example:
        - Original weights: {'A': 0.7, 'B': 0.2, 'C': 0.1}
        - limit = 0.5
        - Excess 0.2 from 'A' is redistributed proportionally to 'B' and 'C'
        - Resulting weights: {'A': 0.5, 'B': 0.33, 'C': 0.17}

    Parameters
    ----------
    limit : float
        Maximum weight allowed for any asset.

    Updates
    -------
    temp['weights'] : dict or pd.Series
        Adjusted weights respecting the maximum limit.

    Requires
    --------
    temp['weights'] : dict or pd.Series
        Initial target weights.
    """

    def __init__(self, limit: float = 0.1):
        super().__init__()
        self.limit = limit

    def limit_weights(
        self, weights: Union[dict, pd.Series], limit: float = 0.1
    ) -> pd.Series:
        """
        Limit individual asset weights and redistribute excess proportionally.

        Parameters
        ----------
        weights : dict or pd.Series
            Original weights. Must sum to 1.
        limit : float
            Maximum allowed weight.

        Returns
        -------
        pd.Series
            Adjusted weights with redistributed excess.
        """
        if 1.0 / limit > len(weights):
            raise ValueError("Invalid limit: 1 / limit must be <= number of assets")

        if isinstance(weights, dict):
            weights = pd.Series(weights)

        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1. Current sum: {weights.sum()}")

        res = weights.copy().round(4)
        # Amount to redistribute
        excess = (res[res > limit] - limit).sum()

        # Redistribute to those below the limit
        ok = res[res < limit]
        if not ok.empty:
            ok += (ok / ok.sum()) * excess

        res[res > limit] = limit
        res[res < limit] = ok

        # Recursive check to ensure no weight exceeds the limit
        if any(res > limit):
            return self.limit_weights(res, limit=limit)

        return res

    def __call__(self, target) -> bool:
        """
        Apply weight limits to target.temp['weights'].

        Parameters
        ----------
        target : StrategyBase
            Strategy context. Must have `temp['weights']`.

        Returns
        -------
        bool
            Always True after applying limits.
        """
        tw = target.temp.get("weights", {})
        if not tw:
            return True

        # If limit is smaller than equal weight, zero out all weights
        if self.limit < 1.0 / len(tw):
            target.temp["weights"] = {}
        else:
            target.temp["weights"] = self.limit_weights(tw, self.limit)

        return True
