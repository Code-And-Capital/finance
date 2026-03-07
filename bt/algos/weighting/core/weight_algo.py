from __future__ import annotations

from typing import Any

import pandas as pd

from bt.algos.core import Algo


class WeightAlgo(Algo):
    """Base class for all weight-assigner algos.

    This class standardizes how weighting algos interact with strategy state.
    Implementations are expected to compute a weight mapping and write it to
    ``target.temp['weights']``.

    Shared capabilities include:
    - Normalizing heterogeneous weight payloads to ``dict[str, float]``
    - Writing normalized weights into ``temp``
    - Recording allocation history per timestamp
    """

    def __init__(
        self,
        name: str | None = None,
        track_allocation_history: bool = True,
    ) -> None:
        """Initialize shared weighting state.

        Parameters
        ----------
        name
            Optional algo name.
        track_allocation_history
            Whether to store per-date snapshots in ``self.allocation_history``.
        """
        super().__init__(name=name)
        self.track_allocation_history = track_allocation_history
        self.allocation_history: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _to_weight_dict(
        weights: dict[Any, float] | pd.Series | None,
    ) -> dict[Any, float]:
        """Convert weight payload to ``dict`` and drop missing entries.

        Parameters
        ----------
        weights
            Weight payload as ``dict``, ``Series``, or ``None``.

        Returns
        -------
        dict
            Normalized mapping suitable for ``temp['weights']``.
        """
        if weights is None:
            return {}
        if isinstance(weights, pd.Series):
            return {k: float(v) for k, v in weights.dropna().to_dict().items()}
        if isinstance(weights, dict):
            return {k: float(v) for k, v in weights.items() if pd.notna(v)}
        raise TypeError("weights must be a dict, pandas Series, or None.")

    @staticmethod
    def _normalize_to_one(weights: dict[Any, float]) -> dict[Any, float]:
        """Return ``weights`` normalized to sum to 1.

        Returns an empty mapping when input is empty or sums to zero.
        """
        if not weights:
            return {}
        total = float(sum(weights.values()))
        if total == 0.0:
            return {}
        return {name: weight / total for name, weight in weights.items()}

    def _record_allocation_history(
        self,
        now: pd.Timestamp | None,
        allocations: dict[Any, float] | pd.Series | None,
    ) -> None:
        """Record one allocation snapshot for ``now``.

        Notes
        -----
        - No-op when tracking is disabled or ``now`` is missing.
        - When allocation is empty, a row index is still inserted so run
          history remains aligned with execution dates.
        """
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
        """Normalize and write weights to ``temp['weights']``.

        Parameters
        ----------
        temp
            Strategy temporary state mapping.
        weights
            Raw weight payload.
        now
            Evaluation timestamp used when ``record_history=True``.
        record_history
            Whether to append the written weights to allocation history.

        Returns
        -------
        dict
            The normalized weight mapping written to ``temp``.
        """
        weight_map = self._to_weight_dict(weights)
        temp["weights"] = weight_map
        if record_history:
            self._record_allocation_history(now, weight_map)
        return weight_map
