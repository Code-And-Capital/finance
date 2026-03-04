from collections.abc import Iterable
from typing import Any

import pandas as pd

from bt.core.algo_base import Algo
from utils.date_utils import coerce_timestamp, coerce_timestamp_or_none


class RunOnDate(Algo):
    """Trigger only when ``target.now`` matches one of configured dates.

    Parameters
    ----------
    dates : Any
        Date-like scalar or iterable of date-like values accepted by
        :func:`pandas.to_datetime`.

    Notes
    -----
    Membership checks are set-based for predictable O(1) lookups.
    """

    def __init__(self, dates: Any):
        """Initialize date membership trigger.

        Parameters
        ----------
        dates : Any
            Date-like scalar or iterable of date-like values.

        Raises
        ------
        ValueError
            If any configured date cannot be parsed into a valid timestamp.
        """
        super().__init__()
        raw_dates = self._coerce_dates(dates)
        parsed_dates: list[pd.Timestamp] = []
        for raw in raw_dates:
            parsed_dates.append(coerce_timestamp(raw, "RunOnDate `dates` entry"))
        self._dates = frozenset(parsed_dates)

    def _coerce_dates(self, dates: Any) -> list[Any]:
        if isinstance(dates, (str, pd.Timestamp)) or not isinstance(dates, Iterable):
            return [dates]
        return list(dates)

    def __call__(self, target: Any) -> bool:
        """Evaluate whether current target date is configured to run.

        Parameters
        ----------
        target : Any
            Target object expected to expose ``now``.

        Returns
        -------
        bool
            ``True`` when ``target.now`` matches a configured date, else
            ``False``.
        """
        now = coerce_timestamp_or_none(getattr(target, "now", None))
        if now is None:
            return False
        return now in self._dates
