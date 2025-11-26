from bt.core.algo_base import Algo


class WeighEqually(Algo):
    """
    Assign equal weights to all assets listed in ``target.temp["selected"]``.

    This algo is typically used after a selection algorithm determines which
    assets should be included in the portfolio. It computes equal weights
    (1 / n) for all selected assets and stores the resulting mapping in
    ``target.temp["weights"]``.

    Requires:
        - target.temp["selected"]: list of asset identifiers.

    Sets:
        - target.temp["weights"]: dict mapping asset -> equal weight.

    Behavior:
        - If no assets are selected, ``weights`` is set to an empty dict.
        - Always returns True to allow subsequent algos to run.
    """

    def __init__(self) -> None:
        """Initialize an equal-weighting algorithm."""
        super().__init__()

    def __call__(self, target) -> bool:
        """
        Compute and assign equal weights to all selected assets.

        Args:
            target (StrategyBase):
                The strategy context containing the current state, temporary
                data, children, capital, and timestamp.

        Returns:
            bool: Always True, enabling the next algo in the chain to execute.
        """
        selected = target.temp.get("selected", [])

        if not selected:
            target.temp["weights"] = {}
            return True

        weight = 1.0 / len(selected)
        target.temp["weights"] = {asset: weight for asset in selected}

        return True
