import numpy as np
from typing import Any
from bt.algos.weighting.core import WeightAlgo
from utils.math_utils import validate_non_negative, validate_real


class LimitDeltas(WeightAlgo):
    """Limit per-asset weight changes relative to current child weights.

    This modifier clips target deltas:
    ``new_weight - current_child_weight``
    to either a global limit or per-asset limits, then normalizes to sum to 1
    when total weight is positive.
    """

    def __init__(self, limit: float = 0.1) -> None:
        """Initialize delta limiter.

        Parameters
        ----------
        limit
            Non-negative global delta limit applied to all assets.
        """
        super().__init__()
        self.limit = validate_non_negative(validate_real(limit, "limit"), "limit")

    def _limit_deltas(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        """Recursively clip deltas until post-normalization limits hold."""
        tolerance = 1e-12
        all_keys = set(target_weights.keys())
        if not all_keys:
            return {}

        n_assets = float(len(all_keys))
        lower_sum = sum(weight - self.limit for weight in current_weights.values())
        upper_sum = sum(weight + self.limit for weight in current_weights.values())
        relax_from_lower = max(0.0, (lower_sum - 1.0) / n_assets)
        relax_from_upper = max(0.0, (1.0 - upper_sum) / n_assets)
        effective_limit = self.limit + max(relax_from_lower, relax_from_upper)

        def _recurse(candidate: dict[str, float]) -> dict[str, float]:
            clipped: dict[str, float] = {}
            for name in all_keys:
                current_weight = float(current_weights.get(name, 0.0))
                target_weight = float(candidate.get(name, 0.0))
                delta = target_weight - current_weight
                if abs(delta) > effective_limit:
                    clipped[name] = current_weight + float(effective_limit) * float(
                        np.sign(delta)
                    )
                else:
                    clipped[name] = target_weight

            total_weight = float(sum(clipped.values()))
            if total_weight > 0.0:
                clipped = {k: w / total_weight for k, w in clipped.items()}

            violations = [
                abs(clipped[name] - float(current_weights.get(name, 0.0)))
                - effective_limit
                for name in all_keys
            ]
            if all(v <= tolerance for v in violations):
                return clipped

            return _recurse(clipped)

        return _recurse(dict(target_weights))

    def __call__(self, target: Any) -> bool:
        """Apply delta clipping and write updated weights.

        Returns
        -------
        bool
            ``True`` when processed; ``False`` for invalid context.
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return False
        now = self._resolve_now(target)
        children = getattr(target, "children", None)
        if not isinstance(children, dict):
            return False

        raw_weights = temp.get("weights", {})
        target_weights = self._to_weight_dict(raw_weights)
        if not target_weights:
            self._write_weights(temp, {}, now=now, record_history=True)
            return True

        current_weights = {
            name: float(getattr(children.get(name), "weight", 0.0))
            for name in target_weights.keys()
        }
        adjusted = self._limit_deltas(current_weights, target_weights)
        self._write_weights(temp, adjusted, now=now, record_history=True)
        return True


class LimitWeights(WeightAlgo):
    """Cap per-asset weights and redistribute excess proportionally.

    This modifier reads ``target.temp['weights']``, enforces ``weight <= limit``
    for each asset, redistributes excess among uncapped names, and writes the
    adjusted mapping back to ``temp['weights']``.
    """

    def __init__(self, limit: float) -> None:
        """Initialize max-weight limiter.

        Parameters
        ----------
        limit
            Maximum allowed weight for any single asset. Must satisfy
            ``0 < limit <= 1``.
        """
        super().__init__()
        validated_limit = validate_real(limit, "limit")
        if validated_limit <= 0.0 or validated_limit > 1.0:
            raise ValueError("limit must satisfy 0 < limit <= 1.")
        self.limit = validated_limit

    def _limit_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Iteratively cap and redistribute until all names satisfy ``self.limit``."""
        if not weights:
            return {}

        tolerance = 1e-12
        limited = {name: float(weight) for name, weight in weights.items()}

        while True:
            over = {k: v for k, v in limited.items() if v > self.limit + tolerance}
            if not over:
                return limited

            excess = sum(v - self.limit for v in over.values())
            under = {k: v for k, v in limited.items() if v < self.limit - tolerance}
            under_sum = sum(under.values())
            if under_sum <= 0.0:
                return {}

            for name in over:
                limited[name] = self.limit
            for name, weight in under.items():
                limited[name] = weight + (weight / under_sum) * excess

    def __call__(self, target: Any) -> bool:
        """Apply max-weight cap to ``temp['weights']`` and record history.

        Returns
        -------
        bool
            ``True`` when processed, ``False`` for invalid context or payload.
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return False
        now = self._resolve_now(target)

        raw_weights = temp.get("weights", {})
        weight_map = self._to_weight_dict(raw_weights)

        if not weight_map:
            self._write_weights(temp, {}, now=now, record_history=True)
            return True

        # Infeasible cap for this number of names: no valid solution.
        if self.limit < 1.0 / len(weight_map):
            self._write_weights(temp, weight_map, now=now, record_history=True)
            return True

        limited = self._limit_weights(weight_map)
        self._write_weights(temp, limited, now=now, record_history=True)
        return True
