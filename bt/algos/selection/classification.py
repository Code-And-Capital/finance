from typing import Any

import pandas as pd

from bt.utils.selection_utils import intersect_candidates_with_pool
from .base_selection import SelectAll
from utils.list_utils import normalize_string_list


class SelectSector(SelectAll):
    """Select securities whose sector is in a configured sector set.

    Parameters
    ----------
    sectors : list[str] | tuple[str, ...] | str
        Sector names that are eligible for selection.
    sector_data : str | pandas.DataFrame, optional
        Sector source. If string, read via ``target.get_data(sector_data)``.
        If DataFrame, use it directly.
        Defaults to ``"sector_wide"``.

    Notes
    -----
    - Reads sector labels from the configured sector source
      (``pandas.DataFrame``), where index is dates and columns are tickers.
    - Uses the row at ``target.now`` to determine current sector labels.
    - Candidate pool source is ``target.temp['selected']``.
      If missing/empty, candidates are first populated via :class:`SelectAll`.
    - Output order follows the candidate pool order.
    - Returns ``False`` for malformed state/context.
    """

    def __init__(
        self,
        sectors: list[str] | tuple[str, ...] | str,
        sector_data: str | pd.DataFrame = "sector_wide",
    ) -> None:
        """Initialize sector selector."""
        super().__init__()
        normalized = normalize_string_list(sectors, field_name="SelectSector `sectors`")
        if normalized is None:
            raise TypeError("SelectSector `sectors` must be a string or iterable.")
        self.sector_wide: pd.DataFrame | None = None
        self.sector_key: str | None = None
        if isinstance(sector_data, pd.DataFrame):
            self.sector_wide = sector_data
        elif isinstance(sector_data, str):
            self.sector_key = sector_data
        else:
            raise TypeError(
                "SelectSector `sector_data` must be a DataFrame or temp key string."
            )
        self.sectors: set[str] = set(normalized)

    def __call__(self, target: Any) -> bool:
        """Filter candidate pool by configured sectors."""
        resolved = self._resolve_context_and_candidate_pool(
            target,
            lambda: super(SelectSector, self).__call__(target),
        )
        if resolved is None:
            return False
        temp, _, now, candidate_pool = resolved

        resolved_sector = self._resolve_wide_data_row_at_now(
            now=now,
            inline_wide=self.sector_wide,
            wide_key=self.sector_key,
            key_resolver=lambda key: target.get_data(key),
        )
        if resolved_sector is None:
            return False
        _, sector_row = resolved_sector

        candidate_pool = intersect_candidates_with_pool(
            candidate_pool, list(sector_row.index)
        )
        temp["selected"] = [
            name for name in candidate_pool if sector_row.get(name) in self.sectors
        ]
        return True
