"""Base expected-return algo primitives."""

import abc
from typing import Any

import numpy as np
import pandas as pd

from bt.algos.core import Algo
from utils.list_utils import keep_items_in_pool, normalize_to_list


class ExpectedReturns(Algo, metaclass=abc.ABCMeta):
    """Base class for expected-return estimators.

    Subclasses implement :meth:`calculate_expected_returns` and receive a cleaned
    return-history matrix prepared by this base class.

    Execution contract
    ------------------
    On each call, this class:
    1. Resolves ``temp``, ``universe``, and ``now`` from the target.
    2. Reads ``temp["selected"]`` and intersects with universe columns.
    3. Builds a price window ``[now - lookback, now - lag]``.
    4. Converts prices to returns via ``pct_change().iloc[1:]`` and maps
       ``+/-inf`` to ``NaN``.
    5. Calls subclass :meth:`calculate_expected_returns`.
    6. Writes outputs to:
       - ``temp["expected_returns"]``
       - ``temp["selected"]`` synced to expected-return index
    7. Caches expected returns in ``self.expected_return_estimations`` as a
       DataFrame with timestamps on the index and assets on the columns.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Historical lookback window used for expected-return estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        """Initialize expected-returns base settings."""
        super().__init__()
        if not isinstance(lookback, pd.DateOffset):
            raise TypeError("ExpectedReturns `lookback` must be a pandas.DateOffset.")
        if not isinstance(lag, pd.DateOffset):
            raise TypeError("ExpectedReturns `lag` must be a pandas.DateOffset.")
        self.lookback = lookback
        self.lag = lag
        self.expected_return_estimations = pd.DataFrame(dtype=float)

    @abc.abstractmethod
    def calculate_expected_returns(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.Series | None:
        """Compute expected returns for selected assets.

        Parameters
        ----------
        temp : dict[str, Any]
            Strategy temporary storage dictionary.
        universe : pandas.DataFrame
            Price universe with timestamps on index and assets on columns.
        now : pandas.Timestamp
            Evaluation timestamp.
        selected : list[Any]
            Candidate assets after base-level intersection with universe.
        returns_history : pandas.DataFrame
            Return matrix prepared by the base class.

        Returns
        -------
        pandas.Series | None
            Expected-return vector indexed by asset name. Returning ``None``
            signals failure for this call.
        """
        raise NotImplementedError

    def __call__(self, target: Any) -> bool:
        """Compute expected returns for the active selection.

        Parameters
        ----------
        target : Any
            Strategy-like object exposing ``temp``, ``universe``, and ``now``.

        Returns
        -------
        bool
            ``True`` when expected returns were successfully computed and
            written. ``False`` when required context/inputs are unavailable or
            invalid.
        """
        context = self._resolve_temp_universe_now(target)
        if context is None:
            return False
        temp, universe, now = context

        selected = self._resolve_selected_assets(temp, universe)
        if selected is None:
            return False

        returns_history = self._build_returns_history(
            target=target,
            temp=temp,
            universe=universe,
            now=now,
            selected=selected,
        )
        if returns_history.empty:
            return False

        expected_returns = self.calculate_expected_returns(
            temp=temp,
            universe=universe,
            now=now,
            selected=selected,
            returns_history=returns_history,
        )
        if not isinstance(expected_returns, pd.Series):
            return False

        expected_assets = list(expected_returns.index)
        if expected_assets:
            temp["selected"] = keep_items_in_pool(selected, expected_assets)
        else:
            temp["selected"] = []

        temp["expected_returns"] = expected_returns
        values = pd.to_numeric(expected_returns, errors="coerce")
        self.expected_return_estimations.loc[now, values.index] = values.to_numpy(
            dtype=float
        )
        return True

    def _write_empty_expected_returns_payload(self, temp: dict[str, Any]) -> None:
        """Write empty expected-returns payload in canonical format."""
        temp["expected_returns"] = {}

    def _resolve_selected_assets(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
    ) -> list[Any] | None:
        """Resolve and validate selected assets against universe columns."""
        if "selected" not in temp:
            self._write_empty_expected_returns_payload(temp)
            return None
        try:
            selected = normalize_to_list(temp.get("selected"))
        except TypeError:
            return None
        selected = keep_items_in_pool(list(universe.columns), selected or [])
        if not selected:
            self._write_empty_expected_returns_payload(temp)
            return None
        return selected

    def _returns_from_window(
        self,
        universe: pd.DataFrame,
        selected: list[Any],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Build returns history from a price window."""
        try:
            prices = universe.loc[start:end, selected]
        except (TypeError, KeyError, ValueError):
            return pd.DataFrame()
        if prices.empty:
            return pd.DataFrame()
        returns_history = prices.pct_change().iloc[1:]
        returns_history = returns_history.replace([np.inf, -np.inf], np.nan)
        return returns_history

    def _build_returns_history(
        self,
        target: Any,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
    ) -> pd.DataFrame:
        """Build return history for expected-return estimation."""
        end = now - self.lag
        start = now - self.lookback
        return self._returns_from_window(
            universe=universe,
            selected=selected,
            start=start,
            end=end,
        )
