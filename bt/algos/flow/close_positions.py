from typing import Iterable, Union

import pandas as pd

from bt.core.security import SecurityBase
from bt.algos.core import Algo
from utils.date_utils import coerce_timestamp
from utils.dataframe_utils import normalize_date_series, one_column_frame_to_series


class ClosePositionsAfterDates(Algo):
    """Close security positions when configured cutoff dates are reached.

    Parameters
    ----------
    close_dates : str | pandas.DataFrame | pandas.Series
        Either a setup-data key retrievable via ``target.get_data(name)``, or
        a one-column DataFrame/Series indexed by security name with date-like
        values.
    """

    def __init__(self, close_dates: Union[str, pd.DataFrame, pd.Series]) -> None:
        """Initialize closing rule source.

        Notes
        -----
        Date sources are normalized into a canonical timestamp Series once and
        cached. Candidate security names are also cached and recomputed only
        when the child-name set changes.
        """
        super().__init__()
        self._close_dates: pd.Series | None = None
        self._close_name: str | None = None
        self._close_index_names: set[str] = set()
        self._candidate_security_names: tuple[str, ...] = ()
        self._last_child_name_set: frozenset[str] | None = None

        if isinstance(close_dates, pd.Series):
            self._close_dates = normalize_date_series(
                close_dates.copy(), label="close date"
            )
            self._close_index_names = set(self._close_dates.index)
        elif isinstance(close_dates, pd.DataFrame):
            self._close_dates = normalize_date_series(
                one_column_frame_to_series(close_dates), label="close date"
            )
            self._close_index_names = set(self._close_dates.index)
        else:
            self._close_name = close_dates

    def _resolve_close_dates(self, target) -> pd.Series:
        if self._close_dates is not None:
            return self._close_dates

        source = target.get_data(self._close_name)
        if isinstance(source, pd.DataFrame):
            source = one_column_frame_to_series(source)
        elif not isinstance(source, pd.Series):
            raise TypeError(
                "close_dates source must be a pandas Series or one-column DataFrame."
            )

        self._close_dates = normalize_date_series(source, label="close date")
        self._close_index_names = set(self._close_dates.index)
        return self._close_dates

    def _refresh_candidate_names_if_needed(
        self, child_names: Iterable[str], children: dict
    ) -> None:
        name_set = frozenset(child_names)
        if name_set == self._last_child_name_set:
            return

        self._last_child_name_set = name_set
        self._candidate_security_names = tuple(
            name
            for name in name_set
            if name in self._close_index_names
            and isinstance(children[name], SecurityBase)
        )

    def __call__(self, target) -> bool:
        """Close positions for securities with reached cutoff dates.

        Returns
        -------
        bool
            Always ``True``.
        """
        if "closed" not in target.perm:
            target.perm["closed"] = set()

        close_dates_series = self._resolve_close_dates(target)
        now_ts = coerce_timestamp(target.now, "target.now")

        children = target.children
        self._refresh_candidate_names_if_needed(children.keys(), children)
        sec_names = [
            sec_name
            for sec_name in self._candidate_security_names
            if sec_name not in target.perm["closed"]
        ]

        if not sec_names:
            return True

        closing_now = close_dates_series.loc[sec_names] <= now_ts
        names_to_close = list(closing_now[closing_now].index)
        for sec_name in names_to_close:
            target.close(sec_name, update=False)
            target.perm["closed"].add(sec_name)

        if names_to_close:
            target.root.update(target.now)

        return True
