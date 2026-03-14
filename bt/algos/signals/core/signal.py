from typing import Any

import pandas as pd

from bt.algos.core import Algo


class Signal(Algo):
    """Base class for cross-sectional signals that update ``temp['selected']``.

    This class handles common mechanics:
    - resolve evaluation timestamp with optional lag
    - resolve candidate pool from ``temp['selected']`` with universe fallback
    - fetch latest available price snapshot at/behind evaluation time
    - write selected names back to ``temp['selected']`` as a list

    Subclasses implement ``_compute_signal`` and return a boolean-like
    ``pandas.Series`` indexed by ticker.
    """

    def __init__(self) -> None:
        """Initialize signal base."""
        super().__init__()
        self.history = pd.DataFrame(dtype=bool)
        self._history_columns: list[Any] | None = None

    def _update_history(
        self,
        now: pd.Timestamp,
        universe_columns: list[Any],
        mask: pd.Series,
    ) -> None:
        """Persist boolean signal state for the current timestamp."""
        if self._history_columns is None:
            self._history_columns = list(universe_columns)
            self.history = pd.DataFrame(columns=self._history_columns, dtype=bool)

        active = mask.reindex(self._history_columns, fill_value=False).astype(bool)
        self.history.loc[now, :] = active

    def __call__(self, target: Any) -> bool:
        """Compute and store selected tickers in ``target.temp['selected']``."""
        context = self._resolve_temp_universe_now(target)
        if context is None:
            return False
        temp, universe, now = context

        candidate_pool = self._resolve_candidate_pool_with_fallback(
            temp,
            lambda: temp.__setitem__("selected", list(universe.columns)) or True,
            allowed_candidates=list(universe.columns),
        )
        if candidate_pool is None:
            return False

        if not candidate_pool:
            temp["selected"] = []
            self._update_history(now, list(universe.columns), pd.Series(dtype=bool))
            return True

        signal = self._compute_signal(
            target=target,
            temp=temp,
            universe=universe,
            now=now,
            candidate_pool=candidate_pool,
        )
        if not isinstance(signal, pd.Series):
            return False

        signal = signal.reindex(candidate_pool, fill_value=False)
        try:
            mask = signal.astype(bool)
        except (TypeError, ValueError):
            return False

        temp["selected"] = list(mask[mask].index)
        self._update_history(now, list(universe.columns), mask)
        return True

    def _compute_signal(
        self,
        target: Any,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        candidate_pool: list[Any],
    ) -> pd.Series | None:
        """Return boolean signal series indexed by asset names."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _resolve_latest_prices(
        self, universe: pd.DataFrame, now: pd.Timestamp, candidate_pool: list[Any]
    ) -> pd.Series | None:
        """Return latest non-missing prices up to ``now`` for ``candidate_pool``."""
        try:
            price_history = universe.loc[:now, candidate_pool]
        except (TypeError, KeyError, ValueError):
            return None
        if price_history.empty:
            return None
        return price_history.iloc[-1].dropna()
