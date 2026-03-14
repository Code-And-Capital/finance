from typing import Any

import pandas as pd
import numpy as np

from utils.list_utils import keep_items_in_pool
from .base_selection import SelectAll
from utils.math_utils import validate_integer


class Ranking(SelectAll):
    """Shared base class for ranking selectors.

    Notes
    -----
    - Candidate pool source is ``temp['selected']``.
      If missing/empty, candidates are first populated via :class:`SelectAll`.
    - Ranking is always constrained to names already present in
      ``temp['selected']``.
    - Summary stats are stored in ``self.stats`` with columns
      ``TOTAL_NAMES``, ``MEAN``, ``MEDIAN``.
    """

    def __init__(self) -> None:
        """Initialize common ranking state."""
        super().__init__()
        self.stats = pd.DataFrame(columns=["TOTAL_NAMES", "MEAN", "MEDIAN"])

    def _resolve_ranking_context(
        self,
        target: Any,
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.Timestamp, list[Any]] | None:
        """Resolve selection context and candidate pool for ranking algos."""
        return self._resolve_context_and_candidate_pool(
            target,
            lambda: super(Ranking, self).__call__(target),
        )

    def _prepare_ranked_stat(
        self,
        temp: dict[str, Any],
        stat_key: str,
        candidate_pool: list[str] | None = None,
    ) -> pd.Series | None:
        """Return cleaned/ranked-eligible metric series from temp state."""
        stat = temp.get(stat_key)
        if not isinstance(stat, pd.Series):
            return None
        if candidate_pool is None:
            selected = temp.get("selected")
            if not isinstance(selected, list):
                return None
            candidate_pool = selected

        ranked = stat.dropna()
        ranked = ranked[np.isfinite(ranked)]
        ranked = ranked.loc[ranked.index.intersection(candidate_pool)]
        return ranked

    def _append_selection_stats(
        self,
        now: pd.Timestamp,
        selected_stats: pd.Series,
    ) -> None:
        """Append one selection-summary row to ``self.stats``."""
        if selected_stats.empty:
            row = [0, np.nan, np.nan]
        else:
            row = [len(selected_stats), selected_stats.mean(), selected_stats.median()]
        self.stats.loc[now, ["TOTAL_NAMES", "MEAN", "MEDIAN"]] = row

    def _set_selected_and_record_stats(
        self,
        temp: dict[str, Any],
        now: pd.Timestamp,
        selected_names: list[str],
        selected_stats: pd.Series,
    ) -> bool:
        """Store the current selection and append summary stats."""
        self._append_selection_stats(now, selected_stats)
        temp["selected"] = selected_names
        return True

    def _set_empty_selection(
        self,
        temp: dict[str, Any],
        now: pd.Timestamp,
    ) -> bool:
        """Store an empty selection and append empty summary stats."""
        return self._set_selected_and_record_stats(
            temp=temp,
            now=now,
            selected_names=[],
            selected_stats=pd.Series(dtype="float64"),
        )


class SelectN(Ranking):
    """Select top/bottom names by rank metric in ``target.temp[stat_key]``.

    Parameters
    ----------
    n : int | float
        Selection size. ``0 < n < 1`` means percentage, ``n >= 1`` means
        absolute count.
    sort_descending : bool, optional
        If ``True`` (default), highest metric values rank first.
    stat_key : str
        Key in ``target.temp`` containing the ranking ``pandas.Series``.

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
        stat_key: str,
        sort_descending: bool = True,
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

    def __call__(self, target: Any) -> bool:
        """Rank and select names, storing output in ``target.temp['selected']``."""
        resolved = self._resolve_ranking_context(target)
        if resolved is None:
            return False
        temp, _, now, candidate_pool = resolved

        ranked = self._prepare_ranked_stat(temp, self.stat_key, candidate_pool)
        if ranked is None:
            return False

        if ranked.empty:
            return self._set_empty_selection(temp, now)

        ranked = ranked.sort_values(ascending=self.ascending)
        if 0 < self.n < 1:
            keep_n = max(int(self.n * len(ranked)), 1)
        else:
            keep_n = int(self.n)

        top_names = list(ranked.iloc[:keep_n].index)
        top_stats = ranked.loc[top_names]
        return self._set_selected_and_record_stats(temp, now, top_names, top_stats)


class SelectQuantile(Ranking):
    """Select a single rank bucket (n-tile) from a metric series.

    Parameters
    ----------
    n_tiles : int
        Number of quantile buckets. Must be ``>= 2``.
    tile : int
        1-based bucket index to select in ``[1, n_tiles]``.
    stat_key : str
        Temp key containing the ranking metric as ``pandas.Series``.
    sort_descending : bool, optional
        If ``True`` (default), highest values map to tile ``1``.
        If ``False``, lowest values map to tile ``1``.
    Notes
    -----
    - Tile ``1`` is always the best-ranked bucket.
    - Candidate pool source is ``temp['selected']``.
      If missing/empty, candidates are first populated via :class:`SelectAll`.
    - Ranking is always constrained to ``temp['selected']``.
    - NaN and non-finite metric values are excluded before bucketing.
    - Buckets are created by rank position using ``numpy.array_split``.
    - Summary stats for selected bucket are appended to ``self.stats`` with
      the current market-data timestamp on the index and columns
      ``TOTAL_NAMES``, ``MEAN``, ``MEDIAN``.
    - Returns ``False`` when required state is missing or malformed.
    """

    def __init__(
        self,
        n_tiles: int,
        tile: int,
        stat_key: str,
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

    def __call__(self, target: Any) -> bool:
        """Select configured quantile bucket and write to ``temp['selected']``."""
        resolved = self._resolve_ranking_context(target)
        if resolved is None:
            return False
        temp, _, now, candidate_pool = resolved

        ranked = self._prepare_ranked_stat(temp, self.stat_key, candidate_pool)
        if ranked is None:
            return False

        if ranked.empty:
            return self._set_empty_selection(temp, now)

        ranked = ranked.sort_values(ascending=self.ascending)
        positions = np.array_split(np.arange(len(ranked)), self.n_tiles)
        chosen_positions = positions[self.tile - 1]

        selected_names = list(ranked.iloc[chosen_positions].index)
        selected_stats = ranked.loc[selected_names]
        return self._set_selected_and_record_stats(
            temp, now, selected_names, selected_stats
        )


class SectorDoubleSort(Ranking):
    """Select best quantile within each sector from a ranking metric.

    Parameters
    ----------
    n_tiles : int
        Number of per-sector quantile buckets. Must be ``>= 2``.
    stat_key : str
        Temp key containing ranking metric as ``pandas.Series``.
    sort_descending : bool, optional
        If ``True`` (default), highest values are treated as best.
        If ``False``, lowest values are treated as best.
    sector_data : str | pandas.DataFrame, optional
        Sector source. If string, read via ``target.get_data(sector_data)``.
        If DataFrame, use directly. Defaults to ``"sector_wide"``.

    Notes
    -----
    - Tile ``1`` is always the best-ranked bucket.
    - Candidate pool source is ``temp['selected']``.
      If missing/empty, candidates are first populated via :class:`SelectAll`.
    - Ranking is constrained to ``temp['selected']`` (same contract as SelectN).
    - Sector data must be wide DataFrame (dates x tickers); the row at the
      current market-data timestamp is used for sector labels.
    - Within each sector, securities are sorted by metric and split into ``n``
      buckets via ``numpy.array_split``; the best bucket is selected.
    - Summary stats are appended to ``self.stats`` with columns
      ``TOTAL_NAMES``, ``MEAN``, ``MEDIAN``.
    """

    def __init__(
        self,
        n_tiles: int,
        stat_key: str,
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

    def __call__(self, target: Any) -> bool:
        """Select best per-sector quantile and write into ``temp['selected']``."""
        resolved = self._resolve_ranking_context(target)
        if resolved is None:
            return False
        temp, _, now, candidate_pool = resolved

        ranked = self._prepare_ranked_stat(temp, self.stat_key, candidate_pool)
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
            return self._set_empty_selection(temp, now)

        candidate_names = keep_items_in_pool(list(ranked.index), list(sector_row.index))
        if not candidate_names:
            return self._set_empty_selection(temp, now)

        frame = pd.DataFrame(
            {
                self.stat_key: ranked.loc[candidate_names],
                "SECTOR": sector_row.loc[candidate_names],
            }
        ).dropna(subset=["SECTOR"])

        if frame.empty:
            return self._set_empty_selection(temp, now)

        selected_names: list[str] = []
        for _, group in frame.groupby("SECTOR", sort=False):
            ordered = group.sort_values(self.stat_key, ascending=self.ascending)
            positions = np.array_split(np.arange(len(ordered)), self.n_tiles)
            best_positions = positions[0]
            if len(best_positions) == 0:
                continue
            selected_names.extend(list(ordered.iloc[best_positions].index))

        selected_stats = ranked.loc[selected_names]
        return self._set_selected_and_record_stats(
            temp, now, selected_names, selected_stats
        )


class StandardDoubleSort(Ranking):
    """Select the best second-metric quantile within each first-metric bucket.

    Parameters
    ----------
    n_tiles_1 : int
        Number of quantile buckets for the first sorting pass. Must be ``>= 2``.
    n_tiles_2 : int
        Number of quantile buckets for the second sorting pass. Must be ``>= 2``.
    stat_key_1 : str
        Temp key containing the first-pass ranking metric as ``pandas.Series``.
    stat_key_2 : str
        Temp key containing the second-pass ranking metric as ``pandas.Series``.
    sort_descending : bool, optional
        If ``True`` (default), highest second-pass values are treated as best.
        If ``False``, lowest second-pass values are treated as best.

    Notes
    -----
    - Tile ``1`` is always the best-ranked bucket.
    - Candidate pool source is ``temp['selected']``.
      If missing/empty, candidates are first populated via :class:`SelectAll`.
    - Ranking is constrained to ``temp['selected']``.
    - The first pass sorts by ``stat_key_1`` in descending order before
      splitting into ``n_tiles_1`` buckets via ``numpy.array_split``.
    - Within each first-pass bucket, securities are sorted again by
      ``stat_key_2`` and split into ``n_tiles_2`` buckets; the best second-pass
      bucket is selected from every first-pass bucket, producing
      ``n_tiles_1 x n_tiles_2`` possible portfolios per date.
    - Summary stats are appended to ``self.stats`` with columns
      ``TOTAL_NAMES``, ``MEAN``, ``MEDIAN`` using the selected ``stat_key_2``
      values.
    """

    def __init__(
        self,
        n_tiles_1: int,
        n_tiles_2: int,
        stat_key_1: str,
        stat_key_2: str,
        sort_descending: bool = True,
    ) -> None:
        """Initialize standard double-sorting selector."""
        super().__init__()

        if not isinstance(sort_descending, bool):
            raise TypeError("StandardDoubleSort `sort_descending` must be a bool.")
        n_tiles_1_val = int(
            validate_integer(n_tiles_1, "StandardDoubleSort `n_tiles_1`")
        )
        n_tiles_2_val = int(
            validate_integer(n_tiles_2, "StandardDoubleSort `n_tiles_2`")
        )
        if n_tiles_1_val < 2:
            raise ValueError("StandardDoubleSort `n_tiles_1` must be >= 2.")
        if n_tiles_2_val < 2:
            raise ValueError("StandardDoubleSort `n_tiles_2` must be >= 2.")
        if not isinstance(stat_key_1, str) or not stat_key_1:
            raise TypeError(
                "StandardDoubleSort `stat_key_1` must be a non-empty string."
            )
        if not isinstance(stat_key_2, str) or not stat_key_2:
            raise TypeError(
                "StandardDoubleSort `stat_key_2` must be a non-empty string."
            )

        self.n_tiles_1 = n_tiles_1_val
        self.n_tiles_2 = n_tiles_2_val
        self.stat_key_1 = stat_key_1
        self.stat_key_2 = stat_key_2
        self.first_pass_ascending = False
        self.second_pass_ascending = not sort_descending

    def __call__(self, target: Any) -> bool:
        """Select the best second-pass bucket from each first-pass bucket."""
        resolved = self._resolve_ranking_context(target)
        if resolved is None:
            return False
        temp, _, now, candidate_pool = resolved

        ranked_1 = self._prepare_ranked_stat(temp, self.stat_key_1, candidate_pool)
        ranked_2 = self._prepare_ranked_stat(temp, self.stat_key_2, candidate_pool)
        if ranked_1 is None or ranked_2 is None:
            return False

        valid_names = keep_items_in_pool(list(ranked_1.index), list(ranked_2.index))
        if not valid_names:
            return self._set_empty_selection(temp, now)

        frame = pd.DataFrame(
            {
                self.stat_key_1: ranked_1.loc[valid_names],
                self.stat_key_2: ranked_2.loc[valid_names],
            }
        )

        ordered_first_pass = frame.sort_values(
            self.stat_key_1, ascending=self.first_pass_ascending
        )
        first_pass_positions = np.array_split(
            np.arange(len(ordered_first_pass)), self.n_tiles_1
        )

        selected_names: list[str] = []
        for bucket_positions in first_pass_positions:
            if len(bucket_positions) == 0:
                continue

            first_bucket = ordered_first_pass.iloc[bucket_positions]
            ordered_second_pass = first_bucket.sort_values(
                self.stat_key_2, ascending=self.second_pass_ascending
            )
            second_pass_positions = np.array_split(
                np.arange(len(ordered_second_pass)), self.n_tiles_2
            )
            best_positions = second_pass_positions[0]
            if len(best_positions) == 0:
                continue
            selected_names.extend(list(ordered_second_pass.iloc[best_positions].index))

        selected_stats = ranked_2.loc[selected_names]
        return self._set_selected_and_record_stats(
            temp, now, selected_names, selected_stats
        )
