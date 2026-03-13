"""Exponentially weighted expected-return estimators."""

from typing import Any

import numpy as np
import pandas as pd

from utils.math_utils import validate_integer, validate_non_negative, validate_real

from .core import ExpectedReturns


class EWMAExpectedReturns(ExpectedReturns):
    """Estimate expected returns via exponentially weighted mean returns.

    Parameters
    ----------
    alpha : float, optional
        EWMA smoothing factor in ``(0, 1]``.
    halflife : int, optional
        Positive half-life used to derive ``alpha`` as
        ``1 - 0.5 ** (1 / halflife)``.
    lookback : pandas.DateOffset, optional
        Historical lookback window used to build returns history.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.

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
                "EWMAExpectedReturns requires exactly one of `alpha` or `halflife`."
            )
        if halflife is not None:
            half = validate_non_negative(
                validate_integer(halflife, "EWMAExpectedReturns `halflife`"),
                "EWMAExpectedReturns `halflife`",
            )
            if half <= 0:
                raise ValueError("EWMAExpectedReturns `halflife` must be > 0.")
            self.alpha = 1.0 - (0.5 ** (1.0 / half))
            return

        alpha_val = validate_real(alpha, "EWMAExpectedReturns `alpha`")
        if alpha_val <= 0.0 or alpha_val > 1.0:
            raise ValueError("EWMAExpectedReturns `alpha` must be in (0, 1].")
        self.alpha = alpha_val

    def calculate_expected_returns(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.Series | None:
        """Return latest EWMA mean of returns for selected assets."""
        selected_returns = returns_history.reindex(columns=selected)
        if selected_returns.empty:
            return pd.Series(dtype=float)
        ewma_mean = selected_returns.ewm(alpha=self.alpha).mean()
        if ewma_mean.empty:
            return pd.Series(index=selected, dtype=float)
        latest = ewma_mean.iloc[-1]
        if not isinstance(latest, pd.Series):
            return pd.Series(index=selected, dtype=float)
        return latest.reindex(selected)


class BlendedExpectedReturn(ExpectedReturns):
    """Blend multiple EWMA expected-return estimates across half-lives.

    For each half-life, this estimator computes the latest EWMA mean return.
    The final expected return is the equal-weight average across all component
    estimates.

    Parameters
    ----------
    halflives : list[int], optional
        Positive EWMA half-lives used to build component estimators.
    use_log_returns : bool, optional
        If ``True`` (default), use ``log1p`` transformed returns prior to EWMA
        estimation.
    lookback : pandas.DateOffset, optional
        Historical lookback window used to build returns history.
    lag : pandas.DateOffset, optional
        Delay between evaluation date and end of estimation window.
    """

    def __init__(
        self,
        halflives: list[int] | None = None,
        use_log_returns: bool = True,
        lookback: pd.DateOffset = pd.DateOffset(years=10),
        lag: pd.DateOffset = pd.DateOffset(days=0),
    ) -> None:
        super().__init__(lookback=lookback, lag=lag)
        if halflives is None:
            halflives = [10, 21, 63, 125, 250]
        if not isinstance(halflives, list) or not halflives:
            raise TypeError(
                "BlendedExpectedReturn `halflives` must be a non-empty list."
            )
        validated: list[int] = []
        for value in halflives:
            half = int(
                validate_integer(value, "BlendedExpectedReturn `halflives` value")
            )
            if (
                validate_non_negative(half, "BlendedExpectedReturn `halflives` value")
                <= 0
            ):
                raise ValueError("BlendedExpectedReturn half-lives must be > 0.")
            validated.append(half)
        if not isinstance(use_log_returns, bool):
            raise TypeError("BlendedExpectedReturn `use_log_returns` must be a bool.")
        self.halflives = validated
        self.use_log_returns = use_log_returns

    def calculate_expected_returns(
        self,
        temp: dict[str, Any],
        universe: pd.DataFrame,
        now: pd.Timestamp,
        selected: list[Any],
        returns_history: pd.DataFrame,
    ) -> pd.Series | None:
        """Return equal-weight blend of latest EWMA means by half-life."""
        selected_returns = returns_history.reindex(columns=selected)
        if selected_returns.empty:
            return pd.Series(dtype=float)

        model_input = selected_returns.astype(float)
        if self.use_log_returns:
            model_input = pd.DataFrame(
                np.log1p(model_input),
                index=model_input.index,
                columns=model_input.columns,
            )
            model_input = (
                model_input.replace([float("inf"), -float("inf")], pd.NA)
                .astype(float)
                .dropna(how="all")
            )
            if model_input.empty:
                return pd.Series(index=selected, dtype=float)

        components: list[pd.Series] = []
        for half in self.halflives:
            ewma_mean = model_input.ewm(halflife=half).mean()
            if ewma_mean.empty:
                continue
            latest = ewma_mean.iloc[-1]
            if isinstance(latest, pd.Series):
                components.append(latest.reindex(selected))

        if not components:
            return pd.Series(index=selected, dtype=float)

        stacked = pd.concat(components, axis=1)
        return stacked.mean(axis=1).reindex(selected)
