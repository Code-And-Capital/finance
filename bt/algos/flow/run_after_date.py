from typing import Any, Union

import pandas as pd

from bt.core.algo_base import Algo
from utils.date_utils import coerce_timestamp, coerce_timestamp_or_none


class RunAfterDate(Algo):
    """Return ``True`` only when the current date is strictly after a threshold.

    Parameters
    ----------
    date : str | pandas.Timestamp | Any
        Date-like value parsed by :func:`pandas.to_datetime`.

    Notes
    -----
    This algo is strictly greater-than. When ``target.now == date``, it
    returns ``False``.
    """

    def __init__(self, date: Union[str, pd.Timestamp, Any]):
        """Initialize the date threshold.

        Parameters
        ----------
        date : str | pandas.Timestamp | Any
            Date-like value converted to ``pandas.Timestamp``.

        Raises
        ------
        ValueError
            If ``date`` cannot be parsed into a valid timestamp.
        """
        super().__init__()
        self.date: pd.Timestamp = coerce_timestamp(date, "RunAfterDate `date`")

    def __call__(self, target: Any) -> bool:
        """Evaluate whether target time is strictly after the configured date.

        Parameters
        ----------
        target : Any
            Target object expected to expose a ``now`` attribute.

        Returns
        -------
        bool
            ``True`` when ``target.now > self.date``, else ``False``.
        """
        now = coerce_timestamp_or_none(getattr(target, "now", None))
        if now is None:
            return False

        return now > self.date
