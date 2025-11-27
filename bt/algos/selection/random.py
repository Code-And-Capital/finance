import random
from typing import List, Optional
from bt.core.algo_base import AlgoStack

import random
from typing import List, Optional


class SelectRandomly(AlgoStack):
    """
    Randomly selects a subset of currently-selected assets.

    This Algo picks up to ``n`` assets at random from ``temp['selected']``.
    If that list does not exist, the entire universe column list is used.

    Typical use-case:
        Used after a selection Algo (e.g., SelectAll, SelectMomentum) to
        construct a purely random benchmark strategy.

    Args:
        n (Optional[int]):
            Maximum number of assets to randomly select. If None, all valid
            assets are retained.
        include_no_data (bool):
            If False, assets with NaN values at ``target.now`` are removed.
        include_negative (bool):
            If False, assets with zero or negative prices at ``target.now`` are removed.

    Sets:
        temp["selected"]: A list of randomly-selected tickers.

    Requires:
        temp["selected"] (optional — defaults to universe)
    """

    def __init__(
        self,
        n: Optional[int] = None,
        include_no_data: bool = False,
        include_negative: bool = False,
    ) -> None:
        """
        Initialize a SelectRandomly algorithm.

        Parameters:
            n (Optional[int]):
                Maximum number of items to randomly select.
                If None, selection is unbounded.
            include_no_data (bool):
                If False, drop tickers with missing data at ``target.now``.
            include_negative (bool):
                If False, drop tickers with negative or zero prices.

        Returns:
            None
        """
        super().__init__()
        self.n = n
        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def __call__(self, target) -> bool:
        """
        Execute the random selection algorithm.

        This method:
        1. Loads the current selection list (or defaults to the full universe).
        2. Filters out tickers with missing or invalid prices, unless disabled.
        3. Randomly samples up to `n` tickers.
        4. Stores the final selection in ``temp['selected']``.

        Parameters:
            target:
                The strategy/target object containing:
                    - `temp`: dict-like storage
                    - `universe`: price DataFrame
                    - `now`: current timestamp

        Returns:
            bool:
                Always returns True to signal that the Algo executed successfully.
        """
        # Load existing selection or default to all tickers
        if "selected" in target.temp:
            sel: List[str] = list(target.temp["selected"])
        else:
            sel = list(target.universe.columns)

        # Filter NaN / invalid prices
        if not self.include_no_data:
            prices = target.universe.loc[target.now, sel].dropna()

            if self.include_negative:
                sel = list(prices.index)
            else:
                sel = list(prices[prices > 0].index)

        # Random sample
        if self.n is not None:
            n = min(self.n, len(sel))
            sel = random.sample(sel, n)

        # Save selection
        target.temp["selected"] = sel
        return True
