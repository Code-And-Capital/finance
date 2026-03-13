from typing import Any

from bt.algos.weighting.core import WeightAlgo
from utils.math_utils import validate_real


class ScaleWeights(WeightAlgo):
    """Scale existing target weights by a constant multiplier.

    This modifier reads ``target.temp['weights']``, multiplies each weight by
    ``scale``, and writes the result back to ``temp['weights']``. It uses the
    shared ``WeightAlgo`` write path so allocation history is recorded in a
    consistent format.
    """

    def __init__(self, scale: float):
        """Initialize the scaling modifier.

        Parameters
        ----------
        scale
            Scalar multiplier applied to every input weight.
        """
        super().__init__()
        self.scale = validate_real(scale, "scale")

    def __call__(self, target: Any) -> bool:
        """Apply weight scaling for the current strategy step.

        Behavior
        --------
        - Returns ``False`` when ``target.temp`` is unavailable/invalid.
        - Treats missing ``temp['weights']`` as empty mapping.
        - Accepts weight payloads as ``dict`` or ``pandas.Series``.
        - Records allocation history for the current ``target.now``.

        Returns
        -------
        bool
            ``True`` when scaling was processed; ``False`` for invalid weight
            payload types.
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return False

        raw_weights = temp.get("weights")
        try:
            weight_map = self._to_weight_dict(raw_weights)
        except TypeError:
            return False

        scaled = {name: weight * self.scale for name, weight in weight_map.items()}
        now = self._resolve_now(target)
        self._write_weights(temp, scaled, now=now, record_history=True)
        return True
