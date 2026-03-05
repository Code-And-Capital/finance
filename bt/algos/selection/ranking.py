from typing import Any

import pandas as pd
import numpy as np

from bt.utils.selection_utils import intersect_candidates_with_pool
from .base_selection import SelectAll
from utils.math_utils import validate_integer


def _prepare_ranked_stat(
    temp: dict[str, Any],
    stat_key: str,
    candidate_pool: list[str] | None = None,
) -> pd.Series | None:
    """Return cleaned/ranked-eligible metric series from temp state."""
    stat = temp.get(stat_key)
    if not isinstance(stat, pd.Series):
        return None
    if candidate_pool is None:
        if "selected" not in temp:
            return None
        candidate_pool = temp["selected"]

    ranked = stat.dropna()
    ranked = ranked[np.isfinite(ranked)]
    ranked = ranked.loc[ranked.index.intersection(candidate_pool)]
    return ranked


def _append_selection_stats(
    stats: pd.DataFrame,
    now: pd.Timestamp,
    selected_stats: pd.Series,
) -> pd.DataFrame:
    """Append one selection-summary row and return updated stats DataFrame."""
    if selected_stats.empty:
        row = [0, np.nan, np.nan]
    else:
        row = [len(selected_stats), selected_stats.mean(), selected_stats.median()]
    stats.loc[now, ["TOTAL_NAMES", "MEAN", "MEDIAN"]] = row
    return stats


class SelectN(SelectAll):
    """Select top/bottom names by rank metric in ``target.temp[stat_key]``.

    Parameters
    ----------
    n : int | float
        Selection size. ``0 < n < 1`` means percentage, ``n >= 1`` means
        absolute count.
    sort_descending : bool, optional
        If ``True`` (default), highest metric values rank first.
    stat_key : str, optional
        Key in ``target.temp`` containing the ranking ``pandas.Series``.
        Defaults to ``"stat"``.

    Notes
    -----
    - For percentage selection, count is computed as
      ``max(floor(n * len(ranked)), 1)``.
    - Candidate pool source is ``temp['selected']``.
      If missing/empty, candidates are first populated via :class:`SelectAll`.
    - Returns ``False`` (without mutating selection) when ``target`` context is
      malformed or ``temp[stat_key]`` is missing/non-Series.
    - Ranking is always limited to names already present in ``temp['selected']``.
    """

    def __init__(
        self,
        n: float | int,
        sort_descending: bool = True,
        stat_key: str = "stat",
    ) -> None:
        """Initialize rank selector.

        Raises
        ------
        TypeError
            If ``n`` is not numeric, if absolute ``n`` is non-integer, or when
            ``sort_descending`` is not ``bool``.
        ValueError
            If ``n <= 0``.
        """
        super().__init__()
        if not isinstance(sort_descending, bool):
            raise TypeError("SelectN `sort_descending` must be a bool.")
        if not isinstance(stat_key, str) or not stat_key:
            raise TypeError("SelectN `stat_key` must be a non-empty string.")

        if isinstance(n, bool) or not isinstance(n, (int, float)):
            raise TypeError("SelectN `n` must be numeric.")
        if n <= 0:
            raise ValueError("SelectN `n` must be > 0.")
        if n >= 1:
            # Absolute-count mode is strict integer to avoid silent truncation.
            n = int(validate_integer(n, "SelectN `n`"))

        self.n = n
        self.ascending = not sort_descending
        self.stat_key = stat_key
        self.stats = pd.DataFrame(columns=["TOTAL_NAMES", "MEAN", "MEDIAN"])

    def __call__(self, target: Any) -> bool:
        """Rank and select names, storing output in ``target.temp['selected']``."""
        resolved = self._resolve_context_and_candidate_pool(
            target,
            lambda: super(SelectN, self).__call__(target),
        )
        if resolved is None:
            return False
        temp, _, now, candidate_pool = resolved

        ranked = _prepare_ranked_stat(temp, self.stat_key, candidate_pool)
        if ranked is None:
            return False

        if ranked.empty:
            temp["selected"] = []
            self.stats = _append_selection_stats(self.stats, now, ranked)
            return True

        ranked = ranked.sort_values(ascending=self.ascending)
        if 0 < self.n < 1:
            keep_n = max(int(self.n * len(ranked)), 1)
        else:
            keep_n = int(self.n)

        top_names = list(ranked.iloc[:keep_n].index)
        top_stats = ranked.loc[top_names]
        self.stats = _append_selection_stats(self.stats, now, top_stats)
        temp["selected"] = top_names
        return True


class SelectQuantile(SelectAll):
    """Select a single rank bucket (n-tile) from a metric series.

    Parameters
    ----------
    n_tiles : int
        Number of quantile buckets. Must be ``>= 2``.
    tile : int
        1-based bucket index to select in ``[1, n_tiles]``.
    stat_key : str, optional
        Temp key containing the ranking metric as ``pandas.Series``.
        Defaults to ``"stat"``.
    sort_descending : bool, optional
        If ``True`` (default), highest values map to tile ``1``.
        If ``False``, lowest values map to tile ``1``.
    Notes
    -----
    - Candidate pool source is ``temp['selected']``.
      If missing/empty, candidates are first populated via :class:`SelectAll`.
    - Ranking is always constrained to ``temp['selected']``.
    - NaN and non-finite metric values are excluded before bucketing.
    - Buckets are created by rank position using ``numpy.array_split``.
    - Summary stats for selected bucket are appended to ``self.stats`` with
      index ``target.now`` and columns ``TOTAL_NAMES``, ``MEAN``, ``MEDIAN``.
    - Returns ``False`` when required state is missing or malformed.
    """

    def __init__(
        self,
        n_tiles: int,
        tile: int,
        stat_key: str = "stat",
        sort_descending: bool = True,
    ) -> None:
        """Initialize quantile selector."""
        super().__init__()

        if not isinstance(sort_descending, bool):
            raise TypeError("SelectQuantile `sort_descending` must be a bool.")
        n_tiles_val = int(validate_integer(n_tiles, "SelectQuantile `n_tiles`"))
        tile_val = int(validate_integer(tile, "SelectQuantile `tile`"))
        if n_tiles_val < 2:
            raise ValueError("SelectQuantile `n_tiles` must be >= 2.")
        if tile_val < 1 or tile_val > n_tiles_val:
            raise ValueError(f"SelectQuantile `tile` must be in [1, {n_tiles_val}].")
        if not isinstance(stat_key, str) or not stat_key:
            raise TypeError("SelectQuantile `stat_key` must be a non-empty string.")

        self.n_tiles = n_tiles_val
        self.tile = tile_val
        self.stat_key = stat_key
        self.ascending = not sort_descending
        self.stats = pd.DataFrame(columns=["TOTAL_NAMES", "MEAN", "MEDIAN"])

    def __call__(self, target: Any) -> bool:
        """Select configured quantile bucket and write to ``temp['selected']``."""
        resolved = self._resolve_context_and_candidate_pool(
            target,
            lambda: super(SelectQuantile, self).__call__(target),
        )
        if resolved is None:
            return False
        temp, _, now, candidate_pool = resolved

        ranked = _prepare_ranked_stat(temp, self.stat_key, candidate_pool)
        if ranked is None:
            return False

        if ranked.empty:
            temp["selected"] = []
            self.stats = _append_selection_stats(self.stats, now, ranked)
            return True

        ranked = ranked.sort_values(ascending=self.ascending)
        positions = np.array_split(np.arange(len(ranked)), self.n_tiles)
        chosen_positions = positions[self.tile - 1]

        selected_names = list(ranked.iloc[chosen_positions].index)
        selected_stats = ranked.loc[selected_names]

        self.stats = _append_selection_stats(self.stats, now, selected_stats)

        temp["selected"] = selected_names
        return True


class SectorDoubleSort(SelectAll):
    """Select best quantile within each sector from a ranking metric.

    Parameters
    ----------
    n_tiles : int
        Number of per-sector quantile buckets. Must be ``>= 2``.
    stat_key : str, optional
        Temp key containing ranking metric as ``pandas.Series``.
        Defaults to ``"stat"``.
    sort_descending : bool, optional
        If ``True`` (default), highest values are treated as best.
        If ``False``, lowest values are treated as best.
    sector_data : str | pandas.DataFrame, optional
        Sector source. If string, read via ``target.get_data(sector_data)``.
        If DataFrame, use directly. Defaults to ``"sector_wide"``.

    Notes
    -----
    - Candidate pool source is ``temp['selected']``.
      If missing/empty, candidates are first populated via :class:`SelectAll`.
    - Ranking is constrained to ``temp['selected']`` (same contract as SelectN).
    - Sector data must be wide DataFrame (dates x tickers); row at ``target.now``
      is used for sector labels.
    - Within each sector, securities are sorted by metric and split into ``n``
      buckets via ``numpy.array_split``; the best bucket is selected.
    - Summary stats are appended to ``self.stats`` with columns
      ``TOTAL_NAMES``, ``MEAN``, ``MEDIAN``.
    """

    def __init__(
        self,
        n_tiles: int,
        stat_key: str = "stat",
        sort_descending: bool = True,
        sector_data: str | pd.DataFrame = "sector_wide",
    ) -> None:
        """Initialize double-sorting selector."""
        super().__init__()

        if not isinstance(sort_descending, bool):
            raise TypeError("SectorDoubleSort `sort_descending` must be a bool.")
        n_tiles_val = int(validate_integer(n_tiles, "SectorDoubleSort `n_tiles`"))
        if n_tiles_val < 2:
            raise ValueError("SectorDoubleSort `n_tiles` must be >= 2.")
        if not isinstance(stat_key, str) or not stat_key:
            raise TypeError("SectorDoubleSort `stat_key` must be a non-empty string.")
        if isinstance(sector_data, pd.DataFrame):
            self.sector_wide: pd.DataFrame | None = sector_data
            self.sector_key: str | None = None
        elif isinstance(sector_data, str):
            self.sector_wide = None
            self.sector_key = sector_data
        else:
            raise TypeError(
                "SectorDoubleSort `sector_data` must be a DataFrame or temp key string."
            )

        self.n_tiles = n_tiles_val
        self.stat_key = stat_key
        self.ascending = not sort_descending
        self.stats = pd.DataFrame(columns=["TOTAL_NAMES", "MEAN", "MEDIAN"])

    def __call__(self, target: Any) -> bool:
        """Select best per-sector quantile and write into ``temp['selected']``."""
        resolved = self._resolve_context_and_candidate_pool(
            target,
            lambda: super(SectorDoubleSort, self).__call__(target),
        )
        if resolved is None:
            return False
        temp, _, now, candidate_pool = resolved

        ranked = _prepare_ranked_stat(temp, self.stat_key, candidate_pool)
        if ranked is None:
            return False

        resolved_sector = self._resolve_wide_data_row_at_now(
            now=now,
            inline_wide=self.sector_wide,
            wide_key=self.sector_key,
            key_resolver=lambda key: target.get_data(key),
        )
        if resolved_sector is None:
            return False
        _, sector_row = resolved_sector

        if ranked.empty:
            temp["selected"] = []
            self.stats = _append_selection_stats(self.stats, now, ranked)
            return True

        candidate_names = intersect_candidates_with_pool(
            list(ranked.index), list(sector_row.index)
        )
        if not candidate_names:
            temp["selected"] = []
            self.stats = _append_selection_stats(
                self.stats, now, pd.Series(dtype="float64")
            )
            return True

        frame = pd.DataFrame(
            {
                self.stat_key: ranked.loc[candidate_names],
                "SECTOR": sector_row.loc[candidate_names],
            }
        ).dropna(subset=["SECTOR"])

        if frame.empty:
            temp["selected"] = []
            self.stats = _append_selection_stats(
                self.stats, now, pd.Series(dtype="float64")
            )
            return True

        selected_names: list[str] = []
        for _, group in frame.groupby("SECTOR", sort=False):
            ordered = group.sort_values(self.stat_key, ascending=self.ascending)
            positions = np.array_split(np.arange(len(ordered)), self.n_tiles)
            best_positions = positions[0]
            if len(best_positions) == 0:
                continue
            selected_names.extend(list(ordered.iloc[best_positions].index))

        selected_stats = ranked.loc[selected_names]
        self.stats = _append_selection_stats(self.stats, now, selected_stats)
        temp["selected"] = selected_names
        return True
