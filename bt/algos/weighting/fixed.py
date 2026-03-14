from typing import Any

import pandas as pd
from bt.algos.weighting.core import WeightAlgo


class WeightFixed(WeightAlgo):
    """Assign fixed, user-provided weights.

    This class does not perform optimization. It writes a predefined ticker-to-
    weight mapping into ``target.temp['weights']`` and optionally intersects it
    with ``target.temp['selected']`` when that candidate list is present.
    """

    def __init__(self, **weights: float) -> None:
        """Initialize fixed-weight mapping.

        Parameters
        ----------
        **weights
            Keyword mapping of ``ticker=weight`` entries.
        """
        super().__init__()
        self.weights = {name: float(weight) for name, weight in weights.items()}

    def __call__(self, target: Any) -> bool:
        """Write fixed weights for the current strategy context.

        Returns
        -------
        bool
            ``True`` when weights were written; ``False`` for invalid state.
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return False
        now = self._resolve_now(target)
        weights = dict(self.weights)
        selected_raw = temp.get("selected")
        if selected_raw is not None:
            if not isinstance(selected_raw, list):
                return False
            selected_set = set(selected_raw)
            weights = {
                name: weight for name, weight in weights.items() if name in selected_set
            }
            weights = self._normalize_to_one(weights)
        self._write_weights(temp, weights, now=now, record_history=True)
        return True


class WeightFixedSchedule(WeightAlgo):
    """Assign date-indexed target weights.

    Weight source can be supplied directly as a ``DataFrame`` or loaded from
    ``target.get_data(<key>)``. At the market-data timestamp (``target.last_day``
    when available), the matching row is written to ``target.temp['weights']``
    when available.
    """

    def __init__(self, weights: pd.DataFrame | str) -> None:
        """Initialize target-weight source.

        Parameters
        ----------
        weights
            Either a ``DataFrame`` of target weights indexed by date or a key
            used with ``target.get_data`` to resolve that frame at runtime.
        """
        super().__init__()

        if isinstance(weights, pd.DataFrame):
            self.weight_source = weights
            self.weight_source_key: str | None = None
        elif isinstance(weights, str):
            self.weight_source = None
            self.weight_source_key = weights
        else:
            raise TypeError(
                "WeightFixedSchedule `weights` must be a DataFrame or a string key."
            )

    def __call__(self, target: Any) -> bool:
        """Assign target weights for the current market-data date when available.

        Returns
        -------
        bool
            ``True`` when a row was written, ``False`` otherwise.
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return False
        data_now = self._resolve_market_data_now(target)

        resolved_weights = self._resolve_wide_data_row_at_now(
            now=data_now,
            inline_wide=self.weight_source,
            wide_key=self.weight_source_key,
            key_resolver=lambda key: target.get_data(key),
        )
        if resolved_weights is None:
            return False
        _, row = resolved_weights

        resolved = row.dropna().to_dict()
        selected_raw = temp.get("selected")
        if selected_raw is not None:
            if not isinstance(selected_raw, list):
                return False
            selected_set = set(selected_raw)
            resolved = {
                name: weight
                for name, weight in resolved.items()
                if name in selected_set
            }
            resolved = self._normalize_to_one(resolved)
        self._write_weights(temp, resolved, now=data_now, record_history=True)
        return True
