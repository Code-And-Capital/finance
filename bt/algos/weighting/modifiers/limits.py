import numpy as np
from typing import Any
import pandas as pd
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


class LimitBenchmarkDeviation(WeightAlgo):
    """Limit target-weight deviation from benchmark weights.

    This modifier constrains ``target.temp['weights']`` to stay within
    ``±limit`` around benchmark weights at the current market-data timestamp.
    It uses the same recursive clip+normalize pattern as ``LimitDeltas`` and
    guarantees output sums to 1 by relaxing the effective limit only when
    strict feasibility requires it.
    """

    def __init__(self, limit: float, benchmark_weights: str | pd.DataFrame) -> None:
        """Initialize benchmark-deviation limiter.

        Parameters
        ----------
        limit
            Non-negative max absolute deviation from benchmark per security.
        benchmark_weights
            Either a wide benchmark-weight DataFrame indexed by date, or a key
            resolved via ``target.get_data``.
        """
        super().__init__()
        self.limit = validate_non_negative(validate_real(limit, "limit"), "limit")
        if isinstance(benchmark_weights, pd.DataFrame):
            self.benchmark_source = benchmark_weights
            self.benchmark_source_key: str | None = None
        elif isinstance(benchmark_weights, str):
            self.benchmark_source = None
            self.benchmark_source_key = benchmark_weights
        else:
            raise TypeError(
                "LimitBenchmarkDeviation `benchmark_weights` must be a DataFrame or string key."
            )

    def _limit_deviation(
        self,
        benchmark_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        """Recursively cap benchmark deviation with minimal global relaxation."""
        tolerance = 1e-12
        all_keys = set(benchmark_weights.keys())
        if not all_keys:
            return {}

        n_assets = float(len(all_keys))
        lower_sum = sum(float(benchmark_weights[k]) - self.limit for k in all_keys)
        upper_sum = sum(float(benchmark_weights[k]) + self.limit for k in all_keys)
        relax_from_lower = max(0.0, (lower_sum - 1.0) / n_assets)
        relax_from_upper = max(0.0, (1.0 - upper_sum) / n_assets)
        # Minimal one-shot relaxation needed for global feasibility.
        effective_limit = self.limit + max(relax_from_lower, relax_from_upper)

        def _recurse(candidate: dict[str, float]) -> dict[str, float]:
            clipped: dict[str, float] = {}
            for name in all_keys:
                bench_weight = float(benchmark_weights.get(name, 0.0))
                target_weight = float(candidate.get(name, 0.0))
                deviation = target_weight - bench_weight
                if abs(deviation) > effective_limit:
                    clipped[name] = bench_weight + effective_limit * float(
                        np.sign(deviation)
                    )
                else:
                    clipped[name] = target_weight

            total_weight = float(sum(clipped.values()))
            if total_weight > 0.0:
                clipped = {k: w / total_weight for k, w in clipped.items()}
            else:
                return dict(benchmark_weights)

            violations = [
                abs(clipped[name] - float(benchmark_weights.get(name, 0.0)))
                - effective_limit
                for name in all_keys
            ]
            if all(v <= tolerance for v in violations):
                return clipped

            max_change = max(
                abs(clipped[k] - float(candidate.get(k, 0.0))) for k in all_keys
            )
            if max_change <= tolerance:
                return clipped
            return _recurse(clipped)

        return _recurse(dict(target_weights))

    def _prepare_benchmark_inputs(
        self,
        target: Any,
        temp: dict[str, Any],
        now: pd.Timestamp,
    ) -> dict[str, float] | None:
        """Resolve, filter, and normalize benchmark weights for limiting.

        Returns
        -------
        dict[str, float] | None
            Normalized benchmark weights on success.
            Returns ``{}`` when benchmark universe is empty after filtering.
            Returns ``None`` when resolution/validation fails.
        """
        resolved = self._resolve_wide_data_row_at_now(
            now=now,
            inline_wide=self.benchmark_source,
            wide_key=self.benchmark_source_key,
            key_resolver=lambda key: target.get_data(key),
        )
        if resolved is None:
            return None
        _, benchmark_row = resolved
        if not isinstance(benchmark_row, pd.Series):
            return None

        benchmark_series = benchmark_row[benchmark_row.notna()].astype(float)
        selected_raw = temp.get("selected")
        if selected_raw is not None:
            if not isinstance(selected_raw, list):
                return None
            benchmark_series = benchmark_series.loc[
                benchmark_series.index.intersection(selected_raw)
            ]
        if benchmark_series.empty:
            return {}

        benchmark_total = float(benchmark_series.sum())
        if benchmark_total <= 0.0:
            benchmark_series = pd.Series(
                1.0 / len(benchmark_series),
                index=benchmark_series.index,
                dtype=float,
            )
        else:
            benchmark_series = benchmark_series / benchmark_total

        return benchmark_series.astype(float).to_dict()

    def __call__(self, target: Any) -> bool:
        """Apply benchmark-deviation constraint and write updated weights."""
        temp = self._resolve_temp(target)
        if temp is None:
            return False
        execution_now = self._resolve_now(target)
        data_now = self._resolve_market_data_now(target)
        if data_now is None:
            return False

        target_weights = self._to_weight_dict(temp.get("weights", {}))
        if not target_weights:
            self._write_weights(temp, {}, now=execution_now, record_history=True)
            return True

        benchmark_weights = self._prepare_benchmark_inputs(target, temp, data_now)
        if benchmark_weights is None:
            return False
        if not benchmark_weights:
            self._write_weights(temp, {}, now=execution_now, record_history=True)
            return True

        expanded_target = {
            name: float(target_weights.get(name, 0.0))
            for name in benchmark_weights.keys()
        }
        adjusted = self._limit_deviation(benchmark_weights, expanded_target)
        self._write_weights(temp, adjusted, now=execution_now, record_history=True)
        return True
