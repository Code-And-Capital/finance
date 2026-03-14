from typing import Any

import pandas as pd

from bt.algos.core import Algo
from bt.algos.factors import TotalReturn
from bt.algos.signals.core import Signal
from utils.list_utils import keep_items_in_pool


class MomentumSignal(Signal):
    """Build a momentum signal from an internal total-return factor and rank selector.

    This signal computes a trailing return cross-section internally via
    :class:`~bt.algos.factors.TotalReturn`, then delegates selection to the
    provided ranking algo and converts that selected subset into a boolean
    signal over the candidate pool.

    Parameters
    ----------
    ranking_algo : Algo
        Ranking selector algo (e.g. ``SelectN``, ``SelectQuantile``,
        ``SectorDoubleSort``) that reads from ``temp`` and writes selected names
        into ``temp['selected']``.
    lookback : pandas.DateOffset, optional
        Lookback window used by the internal ``TotalReturn`` factor.
    lag : pandas.DateOffset, optional
        Lag applied to the internal ``TotalReturn`` factor to avoid look-ahead.
    """

    def __init__(
        self,
        ranking_algo: Algo,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        """Initialize momentum signal."""
        super().__init__()
        if not isinstance(ranking_algo, Algo):
            raise TypeError("MomentumSignal `ranking_algo` must be an Algo instance.")
        if not isinstance(lookback, pd.DateOffset):
            raise TypeError("MomentumSignal `lookback` must be a pandas.DateOffset.")
        if not isinstance(lag, pd.DateOffset):
            raise TypeError("MomentumSignal `lag` must be a pandas.DateOffset.")

        self.ranking_algo = ranking_algo
        self.total_return_algo = TotalReturn(lookback=lookback, lag=lag)
        self._register_factor_stats(
            self.total_return_algo.factor_key,
            self.total_return_algo.stats,
        )
        self._register_factor_coverage(
            self.total_return_algo.factor_key,
            self.total_return_algo.coverage_df,
        )

    def _compute_signal(
        self,
        target: Any,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        candidate_pool: list[Any],
    ) -> pd.Series | None:
        """Run ranking selector and return boolean mask over ``candidate_pool``."""
        temp["selected"] = list(candidate_pool)

        if not self.total_return_algo(target):
            return None

        total_return = temp.get(self.total_return_algo.factor_key)
        if not isinstance(total_return, pd.Series):
            return None

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
