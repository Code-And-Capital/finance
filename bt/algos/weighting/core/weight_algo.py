from __future__ import annotations

from typing import Any

import pandas as pd

from bt.algos.core import Algo


class WeightAlgo(Algo):
    """Base class for weighting algos that write ``temp['weights']``.

    Provides shared helpers for:
    - normalizing weight payloads into plain dictionaries
    - writing weights to ``temp``
    - recording allocation history by timestamp
    """

    def __init__(
        self,
        name: str | None = None,
        track_allocation_history: bool = True,
    ) -> None:
        """Initialize common weighting-algo state."""
        super().__init__(name=name)
        self.track_allocation_history = track_allocation_history
        self.allocation_history: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _to_weight_dict(
        weights: dict[Any, float] | pd.Series | None,
    ) -> dict[Any, float]:
        """Return weights as ``dict[str, float]`` while dropping null entries."""
        if weights is None:
            return {}
        if isinstance(weights, pd.Series):
            return {k: float(v) for k, v in weights.dropna().to_dict().items()}
        if isinstance(weights, dict):
            return {k: float(v) for k, v in weights.items() if pd.notna(v)}
        raise TypeError("weights must be a dict, pandas Series, or None.")

    def _record_allocation_history(
        self,
        now: pd.Timestamp | None,
        allocations: dict[Any, float] | pd.Series | None,
    ) -> None:
        """Record one allocation row at ``now`` when tracking is enabled."""
        if not self.track_allocation_history or now is None:
            return
        allocation_map = self._to_weight_dict(allocations)
        if not allocation_map:
            if now not in self.allocation_history.index:
                self.allocation_history = self.allocation_history.reindex(
                    list(self.allocation_history.index) + [now]
                )
            return
        self.allocation_history.loc[now, list(allocation_map.keys())] = list(
            allocation_map.values()
        )

    def _write_weights(
        self,
        temp: dict[str, Any],
        weights: dict[Any, float] | pd.Series | None,
        now: pd.Timestamp | None = None,
        record_history: bool = False,
    ) -> dict[Any, float]:
        """Normalize and write weights to ``temp['weights']``."""
        weight_map = self._to_weight_dict(weights)
        temp["weights"] = weight_map
        if record_history:
            self._record_allocation_history(now, weight_map)
        return weight_map
