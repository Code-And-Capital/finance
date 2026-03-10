"""Datasource for market prices data."""

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd

from data_loading.base_data_source import BaseDataSource
from handyman.prices import get_prices
from utils.logging import log
from visualization.charts import Line
from visualization.figure import Figure


class PricesDataSource(BaseDataSource):
    """Load, transform, and format historical pricing data."""

    def __init__(
        self,
        *,
        figis: Sequence[str] | str,
        start_date: str | None = None,
        end_date: str | None = None,
        configs_path: str | None = None,
    ) -> None:
        super().__init__()
        self.figis = figis
        self.start_date = start_date
        self.end_date = end_date
        self.configs_path = configs_path
        self.pull_start_date: str | None = None

    def load(self) -> pd.DataFrame:
        """Load prices in long format (DATE, FIGI, ADJ_CLOSE)."""
        effective_start_date = self.start_date
        if self.start_date is not None:
            try:
                effective_start_date = (
                    (pd.Timestamp(self.start_date) - pd.DateOffset(years=1))
                    .date()
                    .isoformat()
                )
            except Exception as exc:  # noqa: BLE001
                raise ValueError("start_date must be parseable as a date") from exc
        self.pull_start_date = effective_start_date
        log(
            "PricesDataSource: loading prices for "
            f"figis_count={len(self.figis) if isinstance(self.figis, (list, tuple, set)) else 1} "
            f"start_date={self.start_date} pull_start_date={self.pull_start_date} "
            f"end_date={self.end_date}",
            type="info",
        )
        return get_prices(
            figis=self.figis,
            start_date=self.pull_start_date,
            end_date=self.end_date,
            configs_path=self.configs_path,
        )

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning and normalization on the original long dataframe."""
        log(f"PricesDataSource: transforming {len(data)} rows", type="info")
        requested = self._requested_figis()
        if data.empty:
            log("PricesDataSource: received empty prices dataframe", type="warning")
            if requested:
                log(
                    "PricesDataSource: no data returned for requested FIGIs: "
                    + ", ".join(requested),
                    type="warning",
                )
            return data.copy()
        required_cols = {"DATE", "FIGI", "ADJ_CLOSE"}
        missing_cols = required_cols.difference(data.columns)
        if missing_cols:
            raise ValueError(
                f"Expected columns {sorted(required_cols)}; missing {sorted(missing_cols)}"
            )
        out = data.copy()
        out["DATE"] = pd.to_datetime(out["DATE"])
        out["FIGI"] = out["FIGI"].astype(str).str.strip().str.upper()
        out = out.sort_values(["DATE", "FIGI"]).reset_index(drop=True)
        found = set(out["FIGI"].dropna().astype(str).str.strip().str.upper().unique())
        missing = sorted(set(requested) - found)
        if missing:
            log(
                "PricesDataSource: missing data for requested FIGIs: "
                + ", ".join(missing),
                type="warning",
            )
        log(
            "PricesDataSource: transformed "
            f"rows={len(out)} unique_figis={out['FIGI'].nunique()}",
            type="info",
        )
        return out

    def format(self, dates: Sequence[pd.Timestamp] | pd.Index | None = None) -> None:
        """Populate ``self.formatted_data`` with wide and long price representations."""
        if self.transformed_data is None:
            raise ValueError("run() must be called before format().")
        data = self.transformed_data
        log(
            f"PricesDataSource: formatting {len(data)} transformed rows",
            type="info",
        )
        outputs: Dict[str, pd.DataFrame] = {"prices_long": data.copy()}
        if data.empty:
            outputs["prices_wide"] = pd.DataFrame()
            self.formatted_data = outputs
            log("PricesDataSource: prices_wide is empty", type="warning")
            return

        wide = data.pivot_table(
            index="DATE",
            columns="FIGI",
            values="ADJ_CLOSE",
            aggfunc="last",
        )
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        wide = self._fill_internal_and_single_trailing_gaps(wide)
        # Rebase each FIGI to 100 for full history.
        rebased_full = (wide.pct_change(fill_method=None).fillna(0) + 1).cumprod() * 100
        # Keep values only where wide data is actually present after controlled fills.
        active_mask_full = wide.notna()
        rebased_full = rebased_full.where(active_mask_full)
        rebased = rebased_full
        if self.start_date is not None:
            start_ts = pd.Timestamp(self.start_date)
            window_wide = wide[wide.index >= start_ts]
            # Rebase again after date filtering so the window also starts at 100.
            rebased = (
                window_wide.pct_change(fill_method=None).fillna(0) + 1
            ).cumprod() * 100
            active_mask_window = window_wide.notna()
            rebased = rebased.where(active_mask_window)
        outputs["prices_wide_full_history"] = rebased_full
        outputs["prices_wide"] = rebased
        last_valid_date = rebased.apply(lambda s: s.last_valid_index())
        last_valid_date = last_valid_date[rebased.isna().iloc[-1]]
        outputs["last_valid_date"] = last_valid_date
        self.formatted_data = outputs
        log(
            "PricesDataSource: generated rebased prices_wide "
            f"shape={rebased.shape} full_history_shape={rebased_full.shape}",
            type="info",
        )

    @staticmethod
    def _fill_internal_and_single_trailing_gaps(wide: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill internal gaps and at most one trailing gap per FIGI.

        Rules per column:
        - Internal gaps are forward-filled.
        - Exactly one trailing NaN is forward-filled.
        - Two or more trailing NaNs remain unfilled.
        """
        if wide.empty:
            return wide

        out = wide.copy()
        row_idx = pd.Series(range(len(out.index)), index=out.index)

        for column in out.columns:
            series = out[column]
            valid_positions = series.notna().to_numpy().nonzero()[0]
            if len(valid_positions) == 0:
                continue
            last_valid_pos = int(valid_positions[-1])
            trailing_nans = len(series) - last_valid_pos - 1

            filled = series.ffill()
            if trailing_nans == 1:
                out[column] = filled
            else:
                out[column] = filled.where(row_idx <= last_valid_pos)

        return out

    def plot_prices(
        self,
        *,
        figis: Sequence[str] | None = None,
        use_full_history: bool = False,
        title: str = "Prices",
        height: int = 500,
    ) -> Figure:
        """Create a line chart for rebased prices.

        Parameters
        ----------
        figis
            Optional FIGI subset to plot. If None, all available columns are used.
        use_full_history
            If True, plot ``prices_wide_full_history``; else plot ``prices_wide``.
        title
            Figure title.
        height
            Figure height in pixels.
        """
        key = "prices_wide_full_history" if use_full_history else "prices_wide"
        prices = self.formatted_data.get(key)
        if prices is None:
            raise ValueError("format() must be called before plot_prices().")
        if prices.empty:
            raise ValueError(f"{key} is empty; nothing to plot.")

        requested = None
        if figis is not None:
            requested = [str(f).strip().upper() for f in figis if str(f).strip()]
            missing = [f for f in requested if f not in prices.columns]
            if missing:
                log(
                    "PricesDataSource: requested plot FIGIs not found: "
                    + ", ".join(missing),
                    type="warning",
                )
            requested = [f for f in requested if f in prices.columns]
            if not requested:
                raise ValueError(
                    "None of the requested FIGIs are present in prices_wide."
                )

        y_cols = requested if requested is not None else list(prices.columns)
        line = Line(prices)
        line.create(x="index", y=y_cols, width=2, mode="lines")
        line.quick_styling(
            x_title="Date",
            y_title="Price",
            selector_buttons=False,
            rangeslider=False,
        )

        fig = Figure(rows=1, cols=1)
        fig.add_chart(line, row=1, col=1)
        fig.layout(title=title, height=height, showlegend=True)
        fig.show()
        return fig
