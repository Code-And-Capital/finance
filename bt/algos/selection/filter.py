from bt.core.algo_base import Algo
import pandas as pd


class SelectHasData(Algo):
    """
    Selection Algo that filters tickers based on historical data availability.

    This Algo selects securities that have at least ``min_count`` valid data
    points within a lookback window. It is useful for algorithms that require
    a minimum amount of historical data before they can be applied.

    If ``min_count`` is not provided, it is estimated based on the given
    `DateOffset` using ``get_num_days_required``.

    Additional filtering can be applied to exclude securities with missing,
    zero, or negative prices on the current date.

    Args:
        lookback (pd.DateOffset): The historical window to use for validation.
            Default is 3 months.
        min_count (int, optional): Minimum required data points within the
            lookback period. If None, an estimate is computed.
        include_no_data (bool, optional): If True, do not filter out tickers that
            have missing data on the current date. Defaults to False.
        include_negative (bool, optional): If True, allow zero or negative prices
            on the current date. Defaults to False.

    Sets:
        temp['selected'] (list[str]): List of tickers satisfying the criteria.

    Example:
        >>> algo = SelectHasData(pd.DateOffset(months=3), min_count=57)
        >>> algo(target)
        >>> target.temp['selected']
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        min_count: int = None,
        include_no_data: bool = False,
        include_negative: bool = False,
    ) -> None:
        """
        Initialize the SelectHasData Algo.

        Args:
            lookback (pd.DateOffset): Lookback period used to gather historical data.
            min_count (int, optional): Required number of data points.
            include_no_data (bool): Whether to include tickers with missing data.
            include_negative (bool): Whether to include negative or zero prices.
        """
        super().__init__()
        self.lookback = lookback

        # Compute default min_count if needed
        if min_count is None:
            min_count = self.get_num_days_required(lookback)

        self.min_count = min_count
        self.include_no_data = include_no_data
        self.include_negative = include_negative

    def get_num_days_required(
        self,
        offset: pd.DateOffset,
        period: str = "d",
        perc_required: float = 0.90,
        annualization_factor: int = 252,
    ) -> int:
        """
        Estimate the number of valid data points required for the given offset.

        This helper function provides an approximate number of expected trading
        days within the lookback period and scales it according to the percentage
        requirement.

        Args:
            offset (pd.DateOffset): Offset representing the lookback period.
            period (str): "d" (daily), "m" (monthly), or "y" (annual).
            perc_required (float): Percentage of the theoretical count required.
            annualization_factor (int): Number of trading days in a year.

        Returns:
            int: Estimated required number of data points.
        """
        anchor = pd.to_datetime("2010-01-01")
        delta = anchor - (anchor - offset)

        # Approximate number of trading days
        days = delta.days * 0.69

        if period == "d":
            req = days * perc_required
        elif period == "m":
            req = (days / 20) * perc_required
        elif period == "y":
            req = (days / annualization_factor) * perc_required
        else:
            raise NotImplementedError(
                "Unsupported period: choose from 'd', 'm', or 'y'."
            )

        return int(req)

    def __call__(self, target) -> bool:
        """
        Select tickers with sufficient historical data in the specified lookback window.

        Args:
            target (Any): Strategy/backtest object containing ``universe``,
                ``now``, and ``temp``.

        Returns:
            bool: Always True.
        """
        # Use previous selection if present, else all universe columns
        selected = target.temp.get("selected", target.universe.columns)

        # Filter out tickers not in the universe
        selected = [t for t in selected if t in target.universe.columns]

        # Slice the lookback window
        start_date = target.now - self.lookback
        filt = target.universe.loc[start_date:, selected]

        # Count non-NaN entries
        cnt = filt.count()
        cnt = cnt[cnt >= self.min_count]

        # Apply "current-data" filters
        current_row = target.universe.loc[target.now, cnt.index]

        if not self.include_no_data:
            cnt = cnt[~current_row.isnull()]
            current_row = current_row.loc[cnt.index]  # sync index

            if not self.include_negative:
                cnt = cnt[current_row > 0]

        target.temp["selected"] = list(cnt.index)
        return True


class SelectActive(Algo):
    """
    Filters temp['selected'] to remove securities that have been marked as
    closed or rolled.

    This is used in workflows where ClosePositionsAfterDates or
    RollPositionsAfterDates have marked certain tickers as inactive.
    Even if such tickers still have prices in the universe, they should
    no longer be selected for allocation or further processing.

    Requires:
        * selected       (in temp)
        * closed, rolled (in perm; sets of tickers)

    Sets:
        * selected
    """

    def __call__(self, target) -> bool:
        """
        Remove tickers from temp['selected'] that appear in perm['closed']
        or perm['rolled'].

        Parameters:
            target: Strategy/backtest object providing temp and perm dicts.

        Returns:
            True
        """
        selected = target.temp["selected"]
        rolled = target.perm.get("rolled", set())
        closed = target.perm.get("closed", set())

        inactive = set.union(rolled, closed)
        active = [s for s in selected if s not in inactive]

        target.temp["selected"] = active
        return True
