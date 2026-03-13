"""Performance analytics for strategy equity curves."""

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _PeriodSeries:
    daily_prices: pd.Series
    monthly_prices: pd.Series
    quarterly_prices: pd.Series
    yearly_prices: pd.Series
    daily_returns: pd.Series
    monthly_returns: pd.Series
    yearly_returns: pd.Series


class TimeSeriesPerformanceStats:
    """Compute performance statistics from a price/equity time series.

    Parameters
    ----------
    prices : pandas.Series
        Strategy price/equity series indexed by datetime.
    rf : float | pandas.Series, optional
        Annual risk-free rate as scalar, or a risk-free price index as Series.
    annualization_factor : int, optional
        Annualization factor used for daily Sharpe/Sortino/vol/mean.
    """

    def __init__(
        self,
        prices: pd.Series,
        rf: Union[float, pd.Series] = 0.0,
        annualization_factor: int = 252,
    ) -> None:
        if not isinstance(prices, pd.Series):
            raise TypeError(
                "TimeSeriesPerformanceStats `prices` must be a pandas Series."
            )
        if prices.empty:
            raise ValueError("TimeSeriesPerformanceStats `prices` cannot be empty.")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError(
                "TimeSeriesPerformanceStats `prices` index must be a DatetimeIndex."
            )
        if annualization_factor <= 0:
            raise ValueError(
                "TimeSeriesPerformanceStats `annualization_factor` must be > 0."
            )

        cleaned = (
            pd.to_numeric(prices, errors="coerce").dropna().astype(float).sort_index()
        )
        if len(cleaned) < 1:
            raise ValueError(
                "TimeSeriesPerformanceStats `prices` has no usable numeric data."
            )

        self.prices = cleaned
        self.name = cleaned.name
        self.rf = rf
        self.annualization_factor = int(annualization_factor)

        self.stats = pd.Series(dtype=float)

        self._update(self.prices)

    def _year_fraction(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        """Return elapsed years between two timestamps using ACT/ACT time basis."""
        if end < start:
            raise ValueError("end must be on or after start.")
        return (end - start).total_seconds() / 31_557_600.0

    def _cagr_from_prices(self, prices: pd.Series) -> float:
        """Return CAGR from a strictly positive price series."""
        if len(prices) < 2:
            return float("nan")
        start = prices.index[0]
        end = prices.index[-1]
        yf = self._year_fraction(start, end)
        if yf <= 0:
            return float("nan")
        first = float(prices.iloc[0])
        last = float(prices.iloc[-1])
        if first <= 0:
            return float("nan")
        return (last / first) ** (1.0 / yf) - 1.0

    def _periodize(self, prices: pd.Series) -> _PeriodSeries:
        """Build normalized periodic prices and return series."""
        daily_prices = prices.resample("D").last().dropna()
        monthly_prices = prices.resample("ME").last().dropna()
        quarterly_prices = prices.resample("QE").last().dropna()
        yearly_prices = prices.resample("YE").last().dropna()

        daily_returns = daily_prices.pct_change().dropna()
        monthly_returns = monthly_prices.pct_change().dropna()
        yearly_returns = yearly_prices.pct_change().dropna()
        return _PeriodSeries(
            daily_prices=daily_prices,
            monthly_prices=monthly_prices,
            quarterly_prices=quarterly_prices,
            yearly_prices=yearly_prices,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns,
        )

    def _period_rf(self, returns: pd.Series, nperiods: int) -> pd.Series:
        """Return periodic risk-free series aligned to returns index."""
        if isinstance(self.rf, pd.Series):
            rf_series = pd.to_numeric(self.rf, errors="coerce").dropna().astype(float)
            if isinstance(rf_series.index, pd.DatetimeIndex):
                rf_returns = rf_series.pct_change().reindex(returns.index)
                return rf_returns.fillna(0.0).astype(float)
            return pd.Series(0.0, index=returns.index, dtype=float)
        scalar = float(self.rf)
        per = (1.0 + scalar) ** (1.0 / float(nperiods)) - 1.0
        return pd.Series(per, index=returns.index, dtype=float)

    @staticmethod
    def _sortino(excess_returns: pd.Series, nperiods: int) -> float:
        """Return annualized Sortino ratio from excess returns."""
        downside = np.minimum(excess_returns.to_numpy(dtype=float), 0.0)
        downside_std = float(np.std(downside, ddof=1))
        if np.isclose(downside_std, 0.0):
            return float("nan")
        return float(excess_returns.mean() / downside_std * np.sqrt(nperiods))

    @staticmethod
    def _sharpe(excess_returns: pd.Series, nperiods: int) -> float:
        """Return annualized Sharpe ratio from excess returns."""
        vol = float(np.std(excess_returns.to_numpy(dtype=float), ddof=1))
        if np.isclose(vol, 0.0):
            return float("nan")
        return float(excess_returns.mean() / vol * np.sqrt(nperiods))

    @staticmethod
    def _drawdown_series(prices: pd.Series) -> pd.Series:
        """Return percentage drawdown series relative to running high-water mark."""
        hwm = prices.cummax()
        return prices / hwm - 1.0

    @staticmethod
    def _max_drawdown_duration(prices: pd.Series) -> int:
        """Return longest consecutive duration (in periods) below high-water mark."""
        if prices.empty:
            return 0
        hwm = prices.cummax()
        underwater = prices < hwm
        max_len = 0
        cur_len = 0
        for is_under in underwater.to_numpy(dtype=bool):
            if is_under:
                cur_len += 1
                if cur_len > max_len:
                    max_len = cur_len
            else:
                cur_len = 0
        return int(max_len)

    def _offset_return(self, prices: pd.Series, offset: pd.DateOffset) -> float:
        """Return trailing return from last observation to ``last - offset``."""
        if prices.empty:
            return float("nan")
        end = prices.index[-1]
        base = prices.loc[: end - offset]
        if base.empty:
            return float("nan")
        denom = float(base.iloc[-1])
        if np.isclose(denom, 0.0):
            return float("nan")
        return float(prices.iloc[-1] / denom - 1.0)

    def _month_quarter_year_to_date(
        self, periods: _PeriodSeries
    ) -> tuple[float, float, float]:
        """Return MTD/QTD/YTD using period-end anchors where available."""
        daily = periods.daily_prices
        monthly = periods.monthly_prices
        quarterly = periods.quarterly_prices
        yearly = periods.yearly_prices

        def _ret(anchor: float) -> float:
            if np.isclose(anchor, 0.0):
                return float("nan")
            return float(daily.iloc[-1] / anchor - 1.0)

        mtd_anchor = (
            float(monthly.iloc[-2]) if len(monthly) > 1 else float(daily.iloc[0])
        )
        qtd_anchor = (
            float(quarterly.iloc[-2]) if len(quarterly) > 1 else float(daily.iloc[0])
        )
        ytd_anchor = float(yearly.iloc[-2]) if len(yearly) > 1 else float(daily.iloc[0])
        return _ret(mtd_anchor), _ret(qtd_anchor), _ret(ytd_anchor)

    def _update(self, obj: pd.Series) -> None:
        """Recompute all performance statistics from a price series."""
        periods = self._periodize(obj)
        prices = periods.daily_prices
        returns = periods.daily_returns
        monthly_returns = periods.monthly_returns
        yearly_returns = periods.yearly_returns

        if len(prices) < 2:
            self.stats = pd.Series(dtype=float)
            return

        has_negative_price = bool((prices < 0.0).any())
        total_return = float(prices.iloc[-1] / prices.iloc[0] - 1.0)
        cagr = 0.0 if has_negative_price else self._cagr_from_prices(prices)
        elapsed_years = self._year_fraction(prices.index[0], prices.index[-1])
        incep = total_return if elapsed_years < 1.0 else cagr

        drawdown = self._drawdown_series(obj)
        max_drawdown = float(drawdown.min()) if not drawdown.empty else float("nan")
        max_drawdown_duration = self._max_drawdown_duration(prices)
        if has_negative_price:
            calmar = 0.0
        else:
            calmar = (
                float(cagr / abs(max_drawdown))
                if not np.isclose(max_drawdown, 0.0)
                else float("nan")
            )

        mtd, qtd, ytd = self._month_quarter_year_to_date(periods)

        three_month = self._offset_return(prices, pd.DateOffset(months=3))
        six_month = self._offset_return(prices, pd.DateOffset(months=6))
        one_year = self._offset_return(prices, pd.DateOffset(years=1))
        three_year = self._offset_return(prices, pd.DateOffset(years=3))
        five_year = self._offset_return(prices, pd.DateOffset(years=5))
        ten_year = self._offset_return(prices, pd.DateOffset(years=10))

        daily_rf = self._period_rf(returns, self.annualization_factor)
        daily_excess = returns - daily_rf
        daily_mean = (
            float(returns.mean() * self.annualization_factor)
            if not returns.empty
            else float("nan")
        )
        daily_vol = (
            float(
                np.std(returns.to_numpy(dtype=float), ddof=1)
                * np.sqrt(self.annualization_factor)
            )
            if len(returns) > 1
            else float("nan")
        )
        daily_sharpe = (
            self._sharpe(daily_excess, self.annualization_factor)
            if len(daily_excess) > 1
            else float("nan")
        )
        daily_sortino = (
            self._sortino(daily_excess, self.annualization_factor)
            if len(daily_excess) > 1
            else float("nan")
        )
        var_95 = float(returns.quantile(0.05)) if not returns.empty else float("nan")
        cvar_95 = (
            float(returns[returns <= var_95].mean())
            if not returns.empty and np.isfinite(var_95)
            else float("nan")
        )
        hit_rate_daily = (
            float((returns > 0).mean()) if not returns.empty else float("nan")
        )

        pos_month_perc = (
            float((monthly_returns > 0).mean())
            if not monthly_returns.empty
            else float("nan")
        )
        avg_up_month = (
            float(monthly_returns[monthly_returns > 0].mean())
            if not monthly_returns.empty
            else float("nan")
        )
        avg_down_month = (
            float(monthly_returns[monthly_returns <= 0].mean())
            if not monthly_returns.empty
            else float("nan")
        )
        win_year_perc = (
            float((yearly_returns > 0).mean())
            if not yearly_returns.empty
            else float("nan")
        )

        rolling_12m_win = float("nan")
        if len(periods.monthly_prices) >= 12:
            r12 = periods.monthly_prices.pct_change(12).dropna()
            if not r12.empty:
                rolling_12m_win = float((r12 > 0).mean())

        self.stats = pd.Series(
            {
                "start": prices.index[0],
                "end": prices.index[-1],
                "total_return": total_return,
                "incep": incep,
                "cagr": cagr,
                "mtd": mtd,
                "qtd": qtd,
                "3m": three_month,
                "6m": six_month,
                "ytd": ytd,
                "1y": one_year,
                "3y": three_year,
                "5y": five_year,
                "10y": ten_year,
                "max_drawdown": max_drawdown,
                "max_drawdown_duration": max_drawdown_duration,
                "calmar": calmar,
                "daily_mean_ann": daily_mean,
                "daily_vol_ann": daily_vol,
                "daily_skew": (
                    float(returns.skew()) if len(returns) > 2 else float("nan")
                ),
                "daily_kurt": (
                    float(returns.kurt()) if len(returns) > 3 else float("nan")
                ),
                "var_95": var_95,
                "cvar_95": cvar_95,
                "hit_rate_daily": hit_rate_daily,
                "sharpe": daily_sharpe,
                "sortino": daily_sortino,
                "best_day": float(returns.max()) if not returns.empty else float("nan"),
                "worst_day": (
                    float(returns.min()) if not returns.empty else float("nan")
                ),
                "best_month": (
                    float(monthly_returns.max())
                    if not monthly_returns.empty
                    else float("nan")
                ),
                "worst_month": (
                    float(monthly_returns.min())
                    if not monthly_returns.empty
                    else float("nan")
                ),
                "best_year": (
                    float(yearly_returns.max())
                    if not yearly_returns.empty
                    else float("nan")
                ),
                "worst_year": (
                    float(yearly_returns.min())
                    if not yearly_returns.empty
                    else float("nan")
                ),
                "avg_up_month": avg_up_month,
                "avg_down_month": avg_down_month,
                "pos_month_perc": pos_month_perc,
                "win_year_perc": win_year_perc,
                "twelve_month_win_perc": rolling_12m_win,
            },
            name=self.name,
        )
