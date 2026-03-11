"""Base covariance algo primitives."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import pandas as pd

from bt.algos.core import Algo
from utils.list_utils import keep_items_in_pool, normalize_to_list


class Covariance(Algo, metaclass=abc.ABCMeta):
    """Base class for covariance estimators.

    Subclasses implement :meth:`calculate_covariance` and receive a cleaned
    return-history matrix prepared by this base class.

    Execution contract
    ------------------
    On each call, this class:
    1. Resolves ``temp``, ``universe``, and ``now`` from the target.
    2. Reads ``temp["selected"]`` and intersects with universe columns.
    3. Builds a price window ``[now - lookback, now - lag]``.
    4. Converts prices to returns via ``pct_change().iloc[1:]`` and maps
       ``+/-inf`` to ``NaN``.
    5. Calls subclass :meth:`calculate_covariance`.
    6. Writes outputs to:
       - ``temp["returns_history"]``
       - ``temp["covariance"]``
       - ``temp["selected"]`` synced to covariance columns
    7. Caches the covariance in ``self.cov_estimations[now]``.

    Parameters
    ----------
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        """Initialize covariance base settings."""
        super().__init__()
        if not isinstance(lookback, pd.DateOffset):
            raise TypeError("Covariance `lookback` must be a pandas.DateOffset.")
        if not isinstance(lag, pd.DateOffset):
            raise TypeError("Covariance `lag` must be a pandas.DateOffset.")
        self.lookback = lookback
        self.lag = lag
        self.cov_estimations: dict[pd.Timestamp, pd.DataFrame] = {}

    @abc.abstractmethod
    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Compute covariance matrix for selected assets.

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
        pandas.DataFrame | None
            Covariance matrix indexed/columned by asset name. Returning
            ``None`` signals failure for this call.
        """
        raise NotImplementedError

    def __call__(self, target: Any) -> bool:
        """Compute covariance for the active selection and persist outputs.

        Parameters
        ----------
        target : Any
            Strategy-like object exposing ``temp``, ``universe``, and ``now``.

        Returns
        -------
        bool
            ``True`` when covariance was successfully computed and written.
            ``False`` when required context/inputs are unavailable or invalid.
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

        covariance = self.calculate_covariance(
            temp=temp,
            universe=universe,
            now=now,
            selected=selected,
            returns_history=returns_history,
        )
        if not isinstance(covariance, pd.DataFrame):
            return False

        covariance_assets = list(covariance.columns)
        if covariance_assets:
            temp["selected"] = keep_items_in_pool(selected, covariance_assets)
        else:
            temp["selected"] = []

        temp["returns_history"] = returns_history
        temp["covariance"] = covariance
        self.cov_estimations[now] = covariance
        return True

    def _write_empty_covariance_payload(self, temp: dict[str, Any]) -> None:
        """Write empty covariance payload in canonical format."""
        temp["covariance"] = {}
        temp["returns_history"] = pd.DataFrame()

    def _resolve_selected_assets(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
    ) -> list[Any] | None:
        """Resolve and validate selected assets against universe columns."""
        if "selected" not in temp:
            self._write_empty_covariance_payload(temp)
            return None
        try:
            selected = normalize_to_list(temp.get("selected"))
        except TypeError:
            return None
        selected = keep_items_in_pool(list(universe.columns), selected or [])
        if not selected:
            self._write_empty_covariance_payload(temp)
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
        """Build return history for covariance estimation."""
        end = now - self.lag
        start = now - self.lookback
        return self._returns_from_window(
            universe=universe,
            selected=selected,
            start=start,
            end=end,
        )

    def _coverage_filtered_fit_data(
        self,
        selected_returns: pd.DataFrame,
        min_coverage: float,
    ) -> tuple[list[Any], pd.DataFrame]:
        """Return eligible assets and complete-case fit data by coverage.

        Parameters
        ----------
        selected_returns : pandas.DataFrame
            Return matrix already restricted to candidate assets.
        min_coverage : float
            Minimum non-missing fraction required per asset in ``[0, 1]``.

        Returns
        -------
        tuple[list[Any], pandas.DataFrame]
            ``(eligible_assets, fit_data)`` where ``fit_data`` contains only
            eligible assets and complete-case rows. If no assets pass coverage,
            both outputs are empty.
        """
        coverage = selected_returns.notna().mean(axis=0)
        eligible_assets = list(coverage[coverage >= min_coverage].index)
        if not eligible_assets:
            return [], pd.DataFrame()
        fit_data = selected_returns[eligible_assets].dropna(how="any")
        return eligible_assets, fit_data
