from bt.core import Algo


class SelectN(Algo):
    """
    Selects the top or bottom N tickers based on the metric stored in temp['stat'].

    This is typically used after another Algo computes a ranking metric. You can
    choose whether to sort descending (e.g., pick highest momentum) or ascending.

    Supports selecting:
        * absolute N (e.g. 10) or
        * percentage N (e.g. 0.2 → select top 20%)

    Args:
        n (int | float):
            Number of items to select.
            - If n >= 1 → selects that many items.
            - If 0 < n < 1 → treated as a percentage of available items.
        sort_descending (bool):
            Sort high-to-low if True, low-to-high if False.
        all_or_none (bool):
            If True, only selects items if full N items are available.
        filter_selected (bool):
            If True, intersect with existing temp['selected'] before ranking.

    Sets:
        temp['selected']

    Requires:
        temp['stat'] : pandas.Series
    """

    def __init__(
        self,
        n: float | int,
        sort_descending: bool = True,
        all_or_none: bool = False,
        filter_selected: bool = True,
    ):
        super().__init__()

        if n <= 0:
            raise ValueError("n must be positive (absolute count or percentage).")

        self.n = n
        self.ascending = not sort_descending
        self.all_or_none = all_or_none
        self.filter_selected = filter_selected

    def __call__(self, target):
        stat = target.temp["stat"].dropna()

        # Optionally filter to previously-selected items
        if self.filter_selected and "selected" in target.temp:
            previously_selected = target.temp["selected"]
            stat = stat.loc[stat.index.intersection(previously_selected)]

        # Nothing to rank
        if stat.empty:
            target.temp["selected"] = []
            return True

        # Sort metric
        stat = stat.sort_values(ascending=self.ascending)

        # Compute number to keep (percentage or absolute)
        if 0 < self.n < 1:
            keep_n = int(round(self.n * len(stat)))
            keep_n = max(keep_n, 1)  # ensure at least one selected
        else:
            keep_n = int(self.n)

        # Perform selection
        selected = list(stat.iloc[:keep_n].index)

        # all_or_none rule
        if self.all_or_none and len(selected) < keep_n:
            selected = []

        target.temp["selected"] = selected
        return True
