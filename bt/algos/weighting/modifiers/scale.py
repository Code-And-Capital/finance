from bt.algos.weighting.core import WeightAlgo


class ScaleWeights(WeightAlgo):
    """
    Algo that scales existing portfolio weights by a constant factor.

    Reads ``target.temp['weights']``, multiplies each value by ``scale``, and
    writes back to ``target.temp['weights']``.

    When weights are missing/empty, this class writes an empty mapping and
    returns ``True``.
    """

    def __init__(self, scale: float):
        """
        Initialize the ScaleWeights algorithm.

        Parameters
        ----------
        scale : float
            The multiplier applied to every current weight. For example:
            - 1.0 → no change
            - 0.5 → reduce exposure by half
            - -1.0 → flip all positions to short
        """
        super().__init__()
        self.scale = scale

    def __call__(self, target) -> bool:
        """
        Apply the scaling factor to the target's existing weights.

        Parameters
        ----------
        target : StrategyBase
            The strategy instance whose temporary weights will be scaled.

        Returns
        -------
        bool
            ``True`` after applying scaling logic.
        """
        weights = target.temp.get("weights")

        # fail gracefully if weights do not exist
        if not weights:
            target.temp["weights"] = {}
            return True

        target.temp["weights"] = {k: w * self.scale for k, w in weights.items()}
        return True
