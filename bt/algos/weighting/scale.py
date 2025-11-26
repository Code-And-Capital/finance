from bt.core.algo_base import Algo


class ScaleWeights(Algo):
    """
    Algo that scales existing portfolio weights by a constant factor.

    This is useful for:
    - Scaling exposures up or down (e.g., 0.5x leverage, 2x leverage)
    - Creating fully short allocations by using a negative scale factor
    - Adjusting weights when working with fixed-income or leverage-aware strategies

    Requires:
        * weights (dict): Must already exist in target.temp["weights"]

    Sets:
        * weights (dict): Updated with scaled values

    Attributes
    ----------
    scale : float
        The factor by which all weights will be multiplied.
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
            True if the operation succeeds.
        """
        weights = target.temp.get("weights")

        # fail gracefully if weights do not exist
        if not weights:
            target.temp["weights"] = {}
            return True

        target.temp["weights"] = {k: w * self.scale for k, w in weights.items()}
        return True
