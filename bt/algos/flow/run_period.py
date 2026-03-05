import abc
from typing import Any

import pandas as pd

from bt.algos.core import Algo
from utils.date_utils import coerce_timestamp


class RunPeriod(Algo, metaclass=abc.ABCMeta):
    """Base class for period-boundary scheduling algos.

    Parameters
    ----------
    run_on_first_date : bool, default True
        Whether to trigger on the first real data date (index position 1).
    run_on_end_of_period : bool, default False
        If ``True``, compare current date to the next index date (period-end
        behavior). If ``False``, compare to the previous date (period-start
        behavior).
    run_on_last_date : bool, default False
        Whether to trigger on the last index date.

    Notes
    -----
    Position ``0`` is treated as a synthetic bootstrap row and never triggers.
    """

    def __init__(
        self,
        run_on_first_date: bool = True,
        run_on_end_of_period: bool = False,
        run_on_last_date: bool = False,
    ):
        super().__init__()
        self.run_on_first_date = run_on_first_date
        self.run_on_end_of_period = run_on_end_of_period
        self.run_on_last_date = run_on_last_date

    def __call__(self, target: Any) -> bool:
        """Determine whether the algo should trigger on current target state.

        Parameters
        ----------
        target : Any
            Target object expected to expose ``now`` and ``data.index``.

        Returns
        -------
        bool
            ``True`` when trigger conditions are met, else ``False``.
        """
        data = getattr(target, "data", None)
        index = getattr(data, "index", None)
        if index is None or len(index) == 0:
            return False

        now_ts = self._resolve_now(target)
        if now_ts is None:
            return False

        current_loc = self._loc_for_timestamp(index, now_ts)
        if current_loc is None:
            return False

        last_loc = len(index) - 1

        if current_loc == 0:
            return False

        if current_loc == 1:
            return self.run_on_first_date

        if current_loc == last_loc:
            return self.run_on_last_date

        offset = 1 if self.run_on_end_of_period else -1
        neighbor_date = index[current_loc + offset]
        neighbor_ts = coerce_timestamp(neighbor_date, "RunPeriod neighbor date")

        return self.compare_dates(now_ts, neighbor_ts)

    def _loc_for_timestamp(self, index: pd.Index, now: pd.Timestamp) -> int | None:
        """Return first index location for ``now`` or ``None`` if absent."""
        locs = [loc for loc in index.get_indexer_for([now]) if loc != -1]
        if not locs:
            return None
        return int(locs[0])

    def compare_dates(self, now: pd.Timestamp, neighbor: pd.Timestamp) -> bool:
        """
        Compare two dates to determine whether a period boundary has occurred.

        Subclasses must implement this method to define period logic
        (e.g., month transition, week transition, quarter transition).

        Parameters
        ----------
        now : pandas.Timestamp
            The current date in the simulation.
        neighbor : pandas.Timestamp
            The date immediately before or after ``now`` depending on context.

        Returns
        -------
        bool
            True if a period boundary has been detected and the algo should run.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("RunPeriod is an abstract base class.")


class RunDaily(RunPeriod):
    """Trigger when calendar day changes across adjacent index rows."""

    def compare_dates(self, now: pd.Timestamp, date_to_compare: pd.Timestamp) -> bool:
        return now.date() != date_to_compare.date()


class RunWeekly(RunPeriod):
    """Trigger when ISO week boundary changes across adjacent rows."""

    def compare_dates(self, now: pd.Timestamp, date_to_compare: pd.Timestamp) -> bool:
        now_iso = now.isocalendar()
        cmp_iso = date_to_compare.isocalendar()

        return now_iso.week != cmp_iso.week


class RunMonthly(RunPeriod):
    """Trigger when month boundary changes across adjacent rows."""

    def compare_dates(self, now: pd.Timestamp, date_to_compare: pd.Timestamp) -> bool:
        return now.month != date_to_compare.month


class RunQuarterly(RunPeriod):
    """Trigger when quarter boundary changes across adjacent rows."""

    def compare_dates(self, now: pd.Timestamp, date_to_compare: pd.Timestamp) -> bool:
        return now.quarter != date_to_compare.quarter


class RunYearly(RunPeriod):
    """Trigger when year boundary changes across adjacent rows."""

    def compare_dates(self, now: pd.Timestamp, date_to_compare: pd.Timestamp) -> bool:
        return now.year != date_to_compare.year
