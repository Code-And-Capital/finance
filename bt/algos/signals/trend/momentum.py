from __future__ import annotations

from typing import Any

import pandas as pd

from bt.algos.core import Algo
from bt.algos.signals.core import Signal
from utils.list_utils import keep_items_in_pool


class MomentumSignal(Signal):
    """Build a momentum signal from a precomputed return factor and rank selector.

    This signal expects a return cross-section to already exist in
    ``temp[total_return_key]`` (for example from ``TotalReturn`` earlier in the
    algo stack). It then delegates selection to the provided ranking algo and
    converts that selected subset into a boolean signal over the candidate pool.

    Parameters
    ----------
    ranking_algo : Algo
        Ranking selector algo (e.g. ``SelectN``, ``SelectQuantile``,
        ``SectorDoubleSort``) that reads from ``temp`` and writes selected names
        into ``temp['selected']``.
    total_return_key : str, optional
        Temp key containing the return metric series. Defaults to
        ``"total_return"``.
    """

    def __init__(
        self,
        ranking_algo: Algo,
        total_return_key: str = "total_return",
    ) -> None:
        """Initialize momentum signal."""
        super().__init__()
        if not isinstance(ranking_algo, Algo):
            raise TypeError("MomentumSignal `ranking_algo` must be an Algo instance.")
        if not isinstance(total_return_key, str) or not total_return_key:
            raise TypeError(
                "MomentumSignal `total_return_key` must be a non-empty string."
            )

        self.ranking_algo = ranking_algo
        self.total_return_key = total_return_key

    def _compute_signal(
        self,
        target: Any,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        candidate_pool: list[Any],
    ) -> pd.Series | None:
        """Run ranking selector and return boolean mask over ``candidate_pool``."""
        total_return = temp.get(self.total_return_key)
        if not isinstance(total_return, pd.Series):
            return None

        temp["selected"] = list(candidate_pool)

        # Ensure ranking algos that default to `stat` consume total return.
        if "stat" not in temp:
            temp["stat"] = total_return

        if not self.ranking_algo(target):
            return None

        ranked_selected_raw = temp.get("selected", [])
        if not isinstance(ranked_selected_raw, list):
            return None
        ranked_selected = keep_items_in_pool(candidate_pool, ranked_selected_raw)

        mask = pd.Series(False, index=candidate_pool, dtype=bool)
        if ranked_selected:
            mask.loc[ranked_selected] = True
        return mask
