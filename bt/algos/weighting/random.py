from bt.core.algo_base import Algo
import random


class WeighRandomly(Algo):
    """
    Algo that assigns random weights to the assets in ``selected``.

    This is primarily useful for benchmarking whether a given weighting
    scheme adds value compared to a naïve random weighting baseline.

    The algorithm uses ``random_weights`` to generate a random weight
    vector subject to optional bounds and an optional target weight sum.

    Parameters
    ----------
    bounds : tuple[float, float], optional
        A `(low, high)` tuple specifying the allowed range for each weight.
        Defaults to `(0.0, 1.0)`.
    weight_sum : float, optional
        Desired sum of weights after randomization.
        Defaults to `1`.

    Sets
    ----
    weights : dict
        Stored in ``target.temp["weights"]`` as a {ticker: weight} mapping.

    Requires
    --------
    selected : list[str]
        Tickers to assign weights to.
    """

    def __init__(self, bounds: tuple = (0.0, 1.0), weight_sum: float = 1):
        """
        Initialize the random weighting algo.

        Parameters
        ----------
        bounds : tuple
            (low, high) bounds for each weight.
        weight_sum : float
            Total desired sum of weights.
        """
        super().__init__()
        self.bounds = bounds
        self.weight_sum = weight_sum

    def random_weights(self, n: int, bounds: tuple = (0.0, 1.0), total: float = 1.0):
        """
        Generate pseudo-random weights that sum to a specified total.

        Produces a list of ``n`` random weights, each constrained within
        the ``bounds`` interval, while ensuring the final weight vector
        sums exactly to ``total``. This is useful for creating random
        benchmark portfolios or random allocation strategies when testing
        robustness.

        The algorithm works by iteratively sampling feasible weight values
        given the remaining number of slots and remaining required total.

        Parameters
        ----------
        n : int
            Number of weights to generate.
        bounds : tuple
            A ``(low, high)`` pair specifying the allowed range for each weight.
        total : float
            Desired total sum of the weights.

        Returns
        -------
        list[float]
            A list of ``n`` random weights that satisfy the bounds and sum constraints.

        Raises
        ------
        ValueError
            If bounds are invalid or it is impossible to reach ``total`` given ``n``.
        """
        low, high = bounds

        if high < low:
            raise ValueError(
                "Upper bound must be greater than or equal to lower bound."
            )

        # Check feasibility: can the bounds accommodate the required total?
        if n * high < total or n * low > total:
            raise ValueError("Solution not possible with given n, bounds, and total.")

        weights = [0.0] * n
        remaining_total = -float(
            total
        )  # Negative target matches original algorithm structure

        for i in range(n):
            remaining = n - i - 1

            remaining_high_sum = remaining * high
            remaining_low_sum = remaining * low

            # Determine feasible range for this weight
            min_allowed = max(-remaining_high_sum - remaining_total, low)
            max_allowed = min(-remaining_low_sum - remaining_total, high)

            # Sample a random weight within bounds
            w_i = random.uniform(min_allowed, max_allowed)
            weights[i] = w_i

            # Update remaining target
            remaining_total += w_i

        # Shuffle to avoid order-based bias
        random.shuffle(weights)

        return weights

    def __call__(self, target) -> bool:
        """
        Generate and assign random weights to the currently selected assets.

        Parameters
        ----------
        target : StrategyBase
            Strategy execution context containing:
            - ``target.temp["selected"]``: list of tickers
            - ``target.temp``: temporary storage dictionary

        Returns
        -------
        bool
            Always True after setting random weights.
        """
        selected = target.temp.get("selected", [])
        n = len(selected)

        # Default to zero weights if no assets selected
        if n == 0:
            target.temp["weights"] = {}
            return True

        try:
            random_vector = self.random_weights(n, self.bounds, self.weight_sum)
            weights = dict(zip(selected, random_vector))
        except Exception:
            # If random generation fails, fall back to equal weights
            fallback = 1.0 / n
            weights = {ticker: fallback for ticker in selected}

        target.temp["weights"] = weights
        return True
