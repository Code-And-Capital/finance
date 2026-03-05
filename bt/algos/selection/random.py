import random
from typing import Any

from .base_selection import SelectAll
from utils.math_utils import validate_integer, validate_non_negative


class SelectRandomly(SelectAll):
    """Randomly select up to ``n`` securities from current candidate set.

    Parameters
    ----------
    n : int
        Maximum number of tickers to select.

    Notes
    -----
    - Sampling is without replacement.
    - If ``n`` exceeds the number of eligible names, all eligible names are
      selected.
    - Result order is randomized and not stable across calls.
    - Returns ``False`` when ``target.temp`` is missing/not dict-like, universe
      is missing/invalid, or ``target.now`` is missing/invalid/not in index.
    """

    def __init__(
        self,
        n: int,
    ) -> None:
        """Initialize random selector.

        Raises
        ------
        TypeError
            If ``n`` is not an integer.
        ValueError
            If ``n`` is negative.
        """
        super().__init__()
        n_val = int(validate_integer(n, "SelectRandomly `n`"))
        self.n = int(validate_non_negative(n_val, "SelectRandomly `n`"))

    def __call__(self, target: Any) -> bool:
        """Compute random selection and store it in ``target.temp['selected']``."""
        resolved = self._resolve_context_and_candidate_pool(
            target,
            lambda: super(SelectRandomly, self).__call__(target),
        )
        if resolved is None:
            return False
        temp, _, _, candidate_pool = resolved

        temp["selected"] = random.sample(
            candidate_pool, min(self.n, len(candidate_pool))
        )
        return True
