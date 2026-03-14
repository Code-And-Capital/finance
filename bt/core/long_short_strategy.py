from copy import deepcopy
from typing import Any

import pandas as pd

from bt.algos.core import AlgoStack
from bt.algos.flow.run_period import RunDaily
from bt.algos.portfolio_ops import Rebalance, RebalanceOverTime
from bt.core.strategy import Strategy
from utils.math_utils import is_zero


class LongShortStrategy(Strategy):
    """
    Executable strategy wrapper built from long and short model sleeves.

    ``LongShortStrategy`` keeps the supplied sleeve strategies outside the
    capital tree. Each sleeve is expected to behave like a model portfolio:
    selection / factor / signal / weighting algos are allowed. If a supplied
    sleeve includes built-in execution algos such as ``Rebalance``, they are
    stripped from the sleeve stack automatically.

    On each run:
    - both sleeves execute their normal algo stacks
    - the wrapper caches each sleeve's latest positive target weights
    - long weights are added and short weights are subtracted
    - the combined signed map is written to ``temp["weights"]``
    - the wrapper's own internally-built stack executes against the real book

    This keeps the existing ``Strategy`` and algo behavior intact while
    providing a clear top-level long/short portfolio primitive.
    """

    def __init__(
        self,
        name: str,
        long_strategy: Strategy,
        short_strategy: Strategy,
        frequency_algo: Any | None = None,
        PAR: float = 100.0,
        long_exposure: float = 1.0,
        short_exposure: float = 1.0,
    ) -> None:
        if not isinstance(long_strategy, Strategy):
            raise TypeError("long_strategy must be a Strategy.")
        if not isinstance(short_strategy, Strategy):
            raise TypeError("short_strategy must be a Strategy.")
        if long_strategy is short_strategy:
            raise ValueError("long_strategy and short_strategy must be different.")

        if frequency_algo is None:
            frequency_algo = RunDaily()

        super().__init__(
            name=name,
            algos=[frequency_algo, Rebalance()],
            PAR=PAR,
        )

        self.long_strategy = deepcopy(long_strategy)
        self.short_strategy = deepcopy(short_strategy)
        self.long_exposure = float(long_exposure)
        self.short_exposure = float(short_exposure)

        if self.long_exposure < 0.0:
            raise ValueError("long_exposure must be >= 0.")
        if self.short_exposure < 0.0:
            raise ValueError("short_exposure must be >= 0.")

        self._sanitize_model_strategy(self.long_strategy)
        self._sanitize_model_strategy(self.short_strategy)

        self._long_weights: dict[str, float] | None = None
        self._short_weights: dict[str, float] | None = None

    @staticmethod
    def _sanitize_model_strategy(strategy: Strategy) -> None:
        """Remove built-in execution algos from a model sleeve stack."""
        sanitized_algos = tuple(
            algo
            for algo in strategy.stack.algos
            if not isinstance(algo, (Rebalance, RebalanceOverTime))
        )
        strategy.stack = AlgoStack(*sanitized_algos)
        strategy.algos = {algo.name: algo for algo in sanitized_algos}

    @staticmethod
    def _normalize_weight_map(
        raw_weights: dict[str, Any] | pd.Series | None,
        *,
        label: str,
    ) -> dict[str, float]:
        """Normalize a sleeve weight mapping into a clean float dict."""
        if raw_weights is None:
            return {}

        if isinstance(raw_weights, pd.Series):
            items = raw_weights.dropna().items()
        elif isinstance(raw_weights, dict):
            items = raw_weights.items()
        else:
            raise TypeError(f"{label} weights must be a dict or pandas Series.")

        normalized: dict[str, float] = {}
        for name, value in items:
            try:
                weight = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{label} weights must contain numeric values. "
                    f"Got {value!r} for {name!r}."
                ) from exc

            if pd.isna(weight) or is_zero(weight):
                continue
            normalized[str(name)] = weight
        return normalized

    def _update_cached_sleeve_weights(self) -> None:
        """Refresh cached sleeve weights from any weights emitted today."""
        if self.long_strategy.bankrupt:
            self._long_weights = {}
        elif "weights" in self.long_strategy.temp:
            self._long_weights = self._normalize_weight_map(
                self.long_strategy.temp.get("weights"),
                label="long_strategy",
            )

        if self.short_strategy.bankrupt:
            self._short_weights = {}
        elif "weights" in self.short_strategy.temp:
            self._short_weights = self._normalize_weight_map(
                self.short_strategy.temp.get("weights"),
                label="short_strategy",
            )

    def _combine_cached_weights(self) -> dict[str, float] | None:
        """Combine the latest long/short sleeve targets into one signed map."""
        if self._long_weights is None and self._short_weights is None:
            return None

        combined: dict[str, float] = {}

        for name, weight in (self._long_weights or {}).items():
            combined[name] = combined.get(name, 0.0) + self.long_exposure * weight

        for name, weight in (self._short_weights or {}).items():
            combined[name] = combined.get(name, 0.0) - self.short_exposure * weight

        return {
            name: weight
            for name, weight in combined.items()
            if not is_zero(weight, tol=1e-12)
        }

    def use_integer_positions(self, integer_positions: bool) -> None:
        """Propagate position policy to both the live book and model sleeves."""
        super().use_integer_positions(integer_positions)
        self.long_strategy.use_integer_positions(integer_positions)
        self.short_strategy.use_integer_positions(integer_positions)

    def set_commissions(self, fn) -> None:
        """Propagate commission function to both the live book and model sleeves."""
        super().set_commissions(fn)
        self.long_strategy.set_commissions(fn)
        self.short_strategy.set_commissions(fn)

    def setup(
        self,
        prices: pd.DataFrame,
        *,
        live_start_date: pd.Timestamp | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize both model sleeves and the live execution book."""
        super().setup(prices, live_start_date=live_start_date, **kwargs)
        self.long_strategy.setup(
            prices,
            live_start_date=live_start_date,
            **kwargs,
        )
        self.short_strategy.setup(
            prices,
            live_start_date=live_start_date,
            **kwargs,
        )
        self._long_weights = None
        self._short_weights = None

    def pre_market_update(self, date: Any, inow: int) -> None:
        """Advance both model sleeves and the live book into pre-market."""
        super().pre_market_update(date, inow)
        if self._is_inactive_bankrupt_child():
            return

        self.long_strategy.pre_market_update(date, inow)
        self.short_strategy.pre_market_update(date, inow)

    def run(self) -> None:
        """Run model sleeves, combine signed weights, then trade the live book."""
        if self.bankrupt:
            self.temp = {}
            self.long_strategy.temp = {}
            self.short_strategy.temp = {}
            return

        self.temp = {}

        self.long_strategy.run()
        self.short_strategy.run()
        self._update_cached_sleeve_weights()

        combined_weights = self._combine_cached_weights()
        if combined_weights is not None:
            self.temp["weights"] = combined_weights
            self.temp["long_weights"] = dict(self._long_weights or {})
            self.temp["short_weights"] = dict(self._short_weights or {})

        self.stack(self)

        for child in self._childrenv:
            child.run()
