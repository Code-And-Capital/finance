import random
from typing import Any

from .base_selection import SelectAll
from bt.utils.selection_utils import (
    resolve_candidate_pool_with_fallback,
    resolve_selection_context,
)
from utils.math_utils import validate_integer, validate_non_negative


class SelectRandomly(SelectAll):
    """Randomly select up to ``n`` securities from current candidate set.

    Parameters
    ----------
    n : int
        Maximum number of tickers to select.
    include_no_data : bool, optional
        If ``False`` (default), exclude securities with missing prices at
        ``target.now``.
    include_negative : bool, optional
        If ``False`` (default), exclude non-positive prices. Ignored when
        ``include_no_data`` is ``True``.

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
        include_no_data: bool = False,
        include_negative: bool = False,
    ) -> None:
        """Initialize random selector.

        Raises
        ------
        TypeError
            If ``n`` is not an integer.
        ValueError
            If ``n`` is negative.
        """
        super().__init__(
            include_no_data=include_no_data, include_negative=include_negative
        )
        n_val = int(validate_integer(n, "SelectRandomly `n`"))
        self.n = int(validate_non_negative(n_val, "SelectRandomly `n`"))

    def __call__(self, target: Any) -> bool:
        """Compute random selection and store it in ``target.temp['selected']``."""
        context = resolve_selection_context(target)
        if context is None:
            return False
        temp, universe, _ = context

        candidate_pool = resolve_candidate_pool_with_fallback(
            temp,
            lambda: super(SelectRandomly, self).__call__(target),
            allowed_candidates=list(universe.columns),
        )
        if candidate_pool is None:
            return False

        temp["selected"] = random.sample(
            candidate_pool, min(self.n, len(candidate_pool))
        )
        return True
