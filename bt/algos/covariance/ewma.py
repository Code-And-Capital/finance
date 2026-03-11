"""Exponentially weighted covariance estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bt.algos.covariance.core import Covariance
from utils.math_utils import validate_integer, validate_non_negative, validate_real


class EWMACovariance(Covariance):
    """Estimate covariance using exponentially weighted moving covariance.

    Parameters
    ----------
    alpha : float, optional
        EWMA smoothing factor in ``(0, 1]``.
    halflife : int, optional
        Positive half-life used to derive ``alpha`` as
        ``1 - 0.5 ** (1 / halflife)``.
    lookback : pandas.DateOffset, optional
        Historical window used to build returns.
    lag : pandas.DateOffset, optional
        Lag applied to the evaluation date.

    Notes
    -----
    Exactly one of ``alpha`` or ``halflife`` must be provided.
    """

    def __init__(
        self,
        alpha: float | None = None,
        halflife: int | None = None,
        lookback: pd.DateOffset = pd.DateOffset(years=10),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        if (alpha is None) == (halflife is None):
            raise ValueError(
                "EWMACovariance requires exactly one of `alpha` or `halflife`."
            )

        if halflife is not None:
            half = validate_non_negative(
                validate_integer(halflife, "EWMACovariance `halflife`"),
                "EWMACovariance `halflife`",
            )
            if half <= 0:
                raise ValueError("EWMACovariance `halflife` must be > 0.")
            self.alpha = 1.0 - (0.5 ** (1.0 / half))
            return

        alpha_val = validate_real(alpha, "EWMACovariance `alpha`")
        if alpha_val <= 0.0 or alpha_val > 1.0:
            raise ValueError("EWMACovariance `alpha` must be in (0, 1].")
        self.alpha = alpha_val

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return EWMA covariance matrix at the latest available timestamp.

        Parameters
        ----------
        temp : dict[str, Any]
            Strategy temporary storage dictionary.
        universe : pandas.DataFrame
            Price universe.
        now : pandas.Timestamp
            Evaluation timestamp.
        selected : list[Any]
            Selected asset names.
        returns_history : pandas.DataFrame
            Arithmetic return matrix from base class preprocessing.
        """
        selected_returns = returns_history.reindex(columns=selected)
        if selected_returns.empty:
            return pd.DataFrame(index=selected, columns=selected, dtype=float)

        ewma_cov = selected_returns.ewm(alpha=self.alpha).cov()
        if ewma_cov.empty:
            return pd.DataFrame(index=selected, columns=selected, dtype=float)

        try:
            latest_ts = ewma_cov.index.get_level_values(0)[-1]
            latest_cov = ewma_cov.xs(latest_ts, level=0)
        except (IndexError, KeyError):
            return pd.DataFrame(index=selected, columns=selected, dtype=float)

        return latest_cov.reindex(index=selected, columns=selected)


class RegimeBlendedCovariance(Covariance):
    """Blend multiple EWMA covariance estimators across half-life regimes.

    This estimator computes one EWMA covariance matrix per half-life pair and
    blends them by simple averaging.

    Parameters
    ----------
    halflife_pairs : list[tuple[int, int]], optional
        Sequence of ``(vol_half_life, cov_half_life)`` pairs.
    min_coverage : float, optional
        Minimum non-missing fraction required for an asset to be included in
        fitting. Must be in ``[0, 1]``.
    use_log_returns : bool, optional
        If ``True`` (default), transform returns via ``np.log1p`` before
        combination fitting.
    lookback : pandas.DateOffset, optional
        Historical lookback window used for covariance estimation.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        halflife_pairs: list[tuple[int, int]] | None = None,
        min_coverage: float = 0.8,
        use_log_returns: bool = True,
        lookback: pd.DateOffset = pd.DateOffset(years=10),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        if halflife_pairs is None:
            halflife_pairs = [(10, 21), (21, 63), (63, 125), (125, 250), (250, 500)]
        if not isinstance(halflife_pairs, list) or not halflife_pairs:
            raise TypeError(
                "RegimeBlendedCovariance `halflife_pairs` must be a non-empty list."
            )
        validated_pairs: list[tuple[int, int]] = []
        for pair in halflife_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    "RegimeBlendedCovariance `halflife_pairs` entries must be (int, int)."
                )
            v_hl = int(validate_integer(pair[0], "RegimeBlendedCovariance half-life"))
            c_hl = int(validate_integer(pair[1], "RegimeBlendedCovariance half-life"))
            if v_hl <= 0 or c_hl <= 0:
                raise ValueError("RegimeBlendedCovariance half-lives must be > 0.")
            validated_pairs.append((v_hl, c_hl))
        coverage = validate_non_negative(
            validate_real(min_coverage, "RegimeBlendedCovariance `min_coverage`"),
            "RegimeBlendedCovariance `min_coverage`",
        )
        if coverage > 1.0:
            raise ValueError("RegimeBlendedCovariance `min_coverage` must be <= 1.")
        if not isinstance(use_log_returns, bool):
            raise TypeError("RegimeBlendedCovariance `use_log_returns` must be a bool.")

        self.halflife_pairs = validated_pairs
        self.min_coverage = coverage
        self.use_log_returns = use_log_returns

    @staticmethod
    def _latest_ewma_covariance(frame: pd.DataFrame, halflife: int) -> pd.DataFrame:
        """Return latest EWMA covariance slice for ``frame``."""
        cov_panel = frame.ewm(halflife=halflife).cov()
        latest_ts = cov_panel.index.get_level_values(0)[-1]
        latest = cov_panel.xs(latest_ts, level=0)
        return latest.reindex(index=frame.columns, columns=frame.columns)

    @classmethod
    def _pair_covariance(
        cls,
        frame: pd.DataFrame,
        vol_half_life: int,
        cov_half_life: int,
    ) -> pd.DataFrame:
        """Build covariance for one ``(vol_half_life, cov_half_life)`` pair."""
        cov_raw = cls._latest_ewma_covariance(frame, cov_half_life).astype(float)
        raw_array = cov_raw.to_numpy(dtype=float)

        raw_var = np.clip(np.diag(raw_array), 0.0, None)
        raw_std = np.sqrt(raw_var)
        denom = np.outer(raw_std, raw_std)
        corr = np.divide(
            raw_array,
            denom,
            out=np.zeros_like(raw_array),
            where=denom > 0,
        )
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        target_var = frame.ewm(halflife=vol_half_life).var().iloc[-1]
        target_std = np.sqrt(np.clip(target_var.to_numpy(dtype=float), 0.0, None))
        cov = corr * np.outer(target_std, target_std)
        return pd.DataFrame(cov, index=frame.columns, columns=frame.columns)

    def calculate_covariance(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Return EWMA-regime blended covariance matrix."""
        selected_returns = returns_history.reindex(columns=selected)
        if selected_returns.empty:
            return pd.DataFrame(index=selected, columns=selected, dtype=float)

        eligible_assets, fit_data = self._coverage_filtered_fit_data(
            selected_returns, self.min_coverage
        )
        if not eligible_assets:
            return pd.DataFrame()
        if fit_data.empty:
            return pd.DataFrame(
                index=eligible_assets, columns=eligible_assets, dtype=float
            )

        model_input = fit_data.astype(float)
        if self.use_log_returns:
            model_input = pd.DataFrame(
                np.log1p(model_input),
                index=model_input.index,
                columns=model_input.columns,
            )
            model_input = model_input.replace([np.inf, -np.inf], np.nan).dropna(
                how="any"
            )
            if model_input.empty:
                return pd.DataFrame(
                    index=eligible_assets, columns=eligible_assets, dtype=float
                )

        if len(eligible_assets) == 1:
            variance = float(model_input.iloc[:, 0].var(ddof=1))
            return pd.DataFrame(
                [[variance]],
                index=eligible_assets,
                columns=eligible_assets,
                dtype=float,
            )

        pair_covariances = [
            self._pair_covariance(model_input, vol_hl, cov_hl)
            for vol_hl, cov_hl in self.halflife_pairs
        ]
        if not pair_covariances:
            return None

        blended = sum(pair_covariances) / float(len(pair_covariances))
        blended = 0.5 * (blended + blended.T)
        return blended.reindex(index=eligible_assets, columns=eligible_assets)
