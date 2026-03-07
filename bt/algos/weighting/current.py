from __future__ import annotations

from typing import Any, Callable

from bt.algos.weighting.core import WeightAlgo
from utils.math_utils import is_zero


class WeightCurrent(WeightAlgo):
    """Use current live child weights after an initial bootstrap weighting.

    On the first run, this assigner executes ``first_weight_algo`` to seed
    ``temp['weights']``. On subsequent runs, it reads non-zero child weights
    from ``target.children`` and writes them into ``temp['weights']``.
    """

    def __init__(self, first_weight_algo: Callable[[Any], bool]) -> None:
        """Initialize with the bootstrap weighting algo.

        Parameters
        ----------
        first_weight_algo
            Callable executed once on the first run to seed initial weights.
        """
        super().__init__()
        if not callable(first_weight_algo):
            raise TypeError("WeightCurrent `first_weight_algo` must be callable.")
        self.first_weight_algo = first_weight_algo
        self.has_run_first = False

    def __call__(self, target: Any) -> bool:
        """Write current weights for the strategy at ``target.now``.

        Returns
        -------
        bool
            ``True`` when weights were written, ``False`` for invalid state.
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return False
        now = self._resolve_now(target)
        selected_raw = temp.get("selected", [])
        if not isinstance(selected_raw, list):
            return False
        selected_set = set(selected_raw)

        if not self.has_run_first:
            if not self.first_weight_algo(target):
                return False
            try:
                initial_weights = self._to_weight_dict(temp.get("weights"))
            except TypeError:
                return False
            self._write_weights(temp, initial_weights, now=now, record_history=True)
            self.has_run_first = True
            return True

        children = getattr(target, "children", None)
        if not isinstance(children, dict):
            return False

        current_weights: dict[str, float] = {}
        for child_name, child in children.items():
            if child_name not in selected_set:
                continue
            child_weight = getattr(child, "weight", 0.0)
            if not is_zero(child_weight):
                current_weights[child_name] = float(child_weight)

        current_weights = self._normalize_to_one(current_weights)
        self._write_weights(temp, current_weights, now=now, record_history=True)
        return True
