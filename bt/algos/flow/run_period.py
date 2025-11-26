import abc
import pandas as pd
from bt.core import Algo
from typing import Any


class RunPeriod(Algo, metaclass=abc.ABCMeta):
    """
    Abstract Algo for triggering logic on specific periodic boundaries.

    This base class allows subclasses to define the meaning of a "period" via
    the abstract method :meth:`compare_dates`. The algo then determines whether
    to run based on:

    - Whether it is the *first* date in the dataset
    - Whether it is the *last* date in the dataset
    - Whether it is the *end* of a period (subclass defines period behavior)
    - Whether it is the *start* of a period (subclass defines period behavior)

    Examples
    --------
    Trigger on end-of-month:
        class RunMonthly(RunPeriod):
            def compare_dates(self, now, next_or_prev):
                return now.month != next_or_prev.month

    Trigger on end-of-week:
        class RunWeekly(RunPeriod):
            def compare_dates(self, now, next_or_prev):
                return now.week != next_or_prev.week

    Parameters
    ----------
    run_on_first_date : bool, default True
        If True, run on the first valid date in the dataset.
    run_on_end_of_period : bool, default False
        If True, run when the current date is the last date of the current period.
    run_on_last_date : bool, default False
        If True, run on the final date of the dataset.

    Notes
    -----
    This class does not define what constitutes a "period." Subclasses must
    implement :meth:`compare_dates` to define the boundaries (e.g., month, week).
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
        """
        Determine whether the algo should run on the current date.

        Parameters
        ----------
        target : bt.backtest.Target
            The current backtest target containing:
            - ``now`` (Timestamp): current simulation date
            - ``data`` (DataFrame/Series): price universe indexed by date

        Returns
        -------
        bool
            True if the algo should execute on this date, False otherwise.
        """
        now = target.now

        # If no date is available, do nothing
        if now is None:
            return False

        index = target.data.index

        # If the date is not part of the dataset, skip
        if now not in index:
            return False

        current_loc = index.get_loc(now)
        last_loc = len(index) - 1

        # Index 0 is an artificial date added by the backtest constructor
        if current_loc == 0:
            return False

        # First valid trading date
        if current_loc == 1:
            return self.run_on_first_date

        # Final date in dataset
        if current_loc == last_loc:
            return self.run_on_last_date

        # --- Period comparison logic ---
        # Convert to Timestamp for attribute access (month/week/quarter/etc.)
        now_ts = pd.Timestamp(now)

        # Determine which neighbor to compare against
        # +1 = end-of-period, -1 = start-of-period
        offset = 1 if self.run_on_end_of_period else -1
        neighbor_date = index[current_loc + offset]
        neighbor_ts = pd.Timestamp(neighbor_date)

        return self.compare_dates(now_ts, neighbor_ts)

    @abc.abstractmethod
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
    """
    Algo that triggers whenever the date changes (day boundary crossed).

    This is typically used for daily-frequency logic, such as daily rebalancing
    or daily indicator updates. The algo compares the current `target.now`
    date to either the previous or next date in the dataset (depending on the
    `run_on_end_of_period` setting inherited from `RunPeriod`).

    Args:
        run_on_first_date (bool, optional):
            If True, the algo returns True the first time it is called.
            Defaults to True.

        run_on_end_of_period (bool, optional):
            If True, the algo determines the date change by comparing with
            *next* available date in the dataset.
            If False, it compares with the *previous* date.
            Defaults to False.

        run_on_last_date (bool, optional):
            If True, the algo returns True on the final date in the dataset.
            Defaults to False.

    Returns:
        bool:
            True if the day changes relative to the comparison date,
            otherwise False.
    """

    def compare_dates(self, now, date_to_compare):
        # Trigger if the day has changed.
        return now.date() != date_to_compare.date()


class RunWeekly(RunPeriod):
    """
    Algo that triggers whenever the ISO week number changes.

    This is typically used for weekly-frequency logic such as weekly rebalancing.
    The algo compares the current `target.now` week number with that of either
    the previous or next date in the dataset (depending on the
    `run_on_end_of_period` setting inherited from :class:`RunPeriod`).

    Args:
        run_on_first_date (bool, optional):
            If True, the algo returns True on the first date in the dataset.
            Defaults to True.

        run_on_end_of_period (bool, optional):
            If True, the algo compares the current date with the *next* date
            in the dataset; otherwise, it compares with the *previous* date.
            Defaults to False.

        run_on_last_date (bool, optional):
            If True, the algo returns True on the final date in the dataset.
            Defaults to False.

    Returns:
        bool:
            True if the ISO week changes relative to the comparison date,
            otherwise False.
    """

    def compare_dates(self, now, date_to_compare):
        now_iso = now.isocalendar()
        cmp_iso = date_to_compare.isocalendar()

        # Trigger when year or week number change
        return (now_iso.year != cmp_iso.year) or (now_iso.week != cmp_iso.week)


class RunMonthly(RunPeriod):
    """
    Algo that triggers whenever the month changes.

    This is typically used for monthly-frequency logic, such as monthly
    rebalancing or reporting. The algo compares the current `target.now` month
    with that of either the previous or next date in the dataset, depending
    on the `run_on_end_of_period` setting inherited from :class:`RunPeriod`.

    Args:
        run_on_first_date (bool, optional):
            If True, the algo returns True on the first date in the dataset.
            Defaults to True.

        run_on_end_of_period (bool, optional):
            If True, the algo compares the current date with the *next* date
            in the dataset; otherwise, it compares with the *previous* date.
            Defaults to False.

        run_on_last_date (bool, optional):
            If True, the algo returns True on the final date in the dataset.
            Defaults to False.

    Returns:
        bool:
            True if the month changes relative to the comparison date,
            otherwise False.
    """

    def compare_dates(self, now, date_to_compare):
        """
        Compare the current date to another date to detect a month change.

        Parameters
        ----------
        now : pandas.Timestamp
            The current date in the backtest.
        date_to_compare : pandas.Timestamp
            The date to compare against (previous or next date in dataset).

        Returns
        -------
        bool
            True if the month or year has changed, False otherwise.
        """
        return (now.year != date_to_compare.year) or (
            now.month != date_to_compare.month
        )


class RunQuarterly(RunPeriod):
    """
    Algo that triggers whenever the quarter changes.

    This is typically used for quarterly-frequency logic, such as
    quarterly rebalancing, reporting, or strategy evaluation. The algo
    compares the current `target.now` quarter with that of either the previous
    or next date in the dataset, depending on the `run_on_end_of_period`
    setting inherited from :class:`RunPeriod`.

    Args:
        run_on_first_date (bool, optional):
            If True, the algo returns True on the first date in the dataset.
            Defaults to True.

        run_on_end_of_period (bool, optional):
            If True, the algo compares the current date with the *next* date
            in the dataset; otherwise, it compares with the *previous* date.
            Defaults to False.

        run_on_last_date (bool, optional):
            If True, the algo returns True on the final date in the dataset.
            Defaults to False.

    Returns:
        bool:
            True if the quarter changes relative to the comparison date,
            otherwise False.
    """

    def compare_dates(self, now, date_to_compare):
        """
        Compare the current date to another date to detect a quarter change.

        Parameters
        ----------
        now : pandas.Timestamp
            The current date in the backtest.
        date_to_compare : pandas.Timestamp
            The date to compare against (previous or next date in dataset).

        Returns
        -------
        bool
            True if the quarter or year has changed, False otherwise.
        """
        return (now.year != date_to_compare.year) or (
            now.quarter != date_to_compare.quarter
        )


class RunYearly(RunPeriod):
    """
    Algo that triggers whenever the year changes.

    This is typically used for yearly-frequency logic, such as annual
    rebalancing, reporting, or strategy evaluation. The algo compares the
    current `target.now` year with that of either the previous or next date
    in the dataset, depending on the `run_on_end_of_period` setting inherited
    from :class:`RunPeriod`.

    Args:
        run_on_first_date (bool, optional):
            If True, the algo returns True on the first date in the dataset.
            Defaults to True.

        run_on_end_of_period (bool, optional):
            If True, the algo compares the current date with the *next* date
            in the dataset; otherwise, it compares with the *previous* date.
            Defaults to False.

        run_on_last_date (bool, optional):
            If True, the algo returns True on the final date in the dataset.
            Defaults to False.

    Returns:
        bool:
            True if the year changes relative to the comparison date,
            otherwise False.
    """

    def compare_dates(self, now, date_to_compare):
        """
        Compare the current date to another date to detect a year change.

        Parameters
        ----------
        now : pandas.Timestamp
            The current date in the backtest.
        date_to_compare : pandas.Timestamp
            The date to compare against (previous or next date in dataset).

        Returns
        -------
        bool
            True if the year has changed, False otherwise.
        """
        return now.year != date_to_compare.year
