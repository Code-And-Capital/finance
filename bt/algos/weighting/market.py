from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bt.algos.weighting.core import WeightAlgo
from bt.algos.weighting.optimizers.base_optimizer import BaseOptimizer
from bt.algos.weighting.optimizers.validators import validate_series


class MarketWeightOptimizer(BaseOptimizer):
    """Analytical market-cap weighting optimizer.

    This optimizer normalizes positive market caps over the selected universe.
    """

    def set_problem(self, market_caps: Any, selected: list[str]) -> None:
        """Validate/store market-cap input for selected names."""
        self.reset()
        validate_series(market_caps, "MarketWeightOptimizer", "market_caps")

        aligned_caps = market_caps.reindex(selected)
        self.set_problem_data(selected=selected, market_caps=aligned_caps)

    def solve_problem(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Compute market-cap-proportional weights."""
        selected = self.problem_data.get("selected", [])
        market_caps = self.problem_data.get("market_caps")

        caps = market_caps.reindex(selected)
        caps = pd.to_numeric(caps, errors="coerce").dropna()
        caps = caps[caps > 0.0]
        if caps.empty:
            weights: dict[str, float] = {}
        else:
            total = float(np.nansum(caps.to_numpy(dtype=float, copy=True)))
            if not np.isfinite(total) or total <= 0.0:
                weights = {}
            else:
                normalized = caps / total
                weights = {name: float(weight) for name, weight in normalized.items()}

        self.weights_ = weights
        self.success = True
        self.status = "optimal"
        self.message = "Solved analytically."
        return self.get_result()


class WeightMarket(WeightAlgo):
    """Assign market-cap-proportional weights for ``temp['selected']``.

    This assigner reads wide market-cap data from ``target.get_data`` at
    ``target.now`` and delegates normalization to ``MarketWeightOptimizer``.
    """

    def __init__(self, market_caps_key: str = "marketcap_wide") -> None:
        """Initialize market-cap weighting assigner.

        Parameters
        ----------
        market_caps_key
            Key used with ``target.get_data`` to fetch market-cap source data.
        """
        super().__init__()
        self.market_caps_key = market_caps_key
        self.optimizer = MarketWeightOptimizer()

    def __call__(self, target: Any) -> bool:
        """Compute and store market-cap weights for current evaluation date."""
        temp = self._resolve_temp(target)
        if temp is None:
            return False
        now = self._resolve_now(target)

        selected_raw = temp.get("selected", [])
        if not isinstance(selected_raw, list):
            return False
        if not selected_raw:
            self._write_weights(temp, {}, now=now, record_history=True)
            return True

        market_caps_data = target.get_data(self.market_caps_key)
        if not isinstance(market_caps_data, pd.DataFrame):
            raise TypeError("WeightMarket market cap data must be a DataFrame.")
        eval_date = now
        if eval_date is None or eval_date not in market_caps_data.index:
            return False
        market_caps = market_caps_data.loc[eval_date]

        self.optimizer.set_problem(market_caps, selected_raw)
        result = self.optimizer.solve_problem()
        weights = result["weights"]
        self._write_weights(temp, weights, now=now, record_history=True)
        return True
