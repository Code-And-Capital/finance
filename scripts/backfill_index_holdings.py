"""Backfill missing index-holdings dates from the latest available snapshot."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

# Ensure repository-root imports work when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bt
from bt.algos.flow import RunOnce
from bt.algos.portfolio_ops import Rebalance
from bt.algos.selection import SelectAll, SelectWhere, SelectActive
from bt.algos.weighting import WeighTarget
from bt.core import Strategy
from bt.engine import Backtest
from sqlalchemy import types as satypes

from connectors.azure_data_source import default_azure_data_source
from handyman.holdings import get_index_holdings
from handyman.prices import get_prices
from utils.logging import log


def _normalize_target_dates(
    target_dates: Sequence[str | pd.Timestamp],
) -> pd.DatetimeIndex:
    """Normalize, validate, and sort target dates."""
    dates = pd.to_datetime(list(target_dates), errors="raise")
    if len(dates) == 0:
        raise ValueError("target_dates cannot be empty")
    return pd.DatetimeIndex(sorted(pd.unique(dates)))


def _build_weight_schedule(
    *,
    base_weights: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create a date-indexed weight matrix using one base snapshot for all target dates."""
    if base_weights.empty:
        raise ValueError("base_weights is empty; cannot build schedule")

    base_row = base_weights.iloc[[0]].copy()
    rows = [base_row.assign(DATE=target_date) for target_date in target_dates]
    scheduled = pd.concat(rows, ignore_index=True).set_index("DATE")
    return scheduled.sort_index()


def build_index_holdings_backfill(
    *,
    index_name: str,
    source_date: str | pd.Timestamp,
    target_dates: Sequence[str | pd.Timestamp],
    write_to_azure: bool = False,
    configs_path: str | None = None,
) -> pd.DataFrame:
    """Build holdings rows for missing dates using latest available index constituents.

    Parameters
    ----------
    index_name
        Index name as stored in holdings table (example: ``"S&P 500"``).
    source_date
        Existing holdings snapshot date used as the base for duplicated weights.
    target_dates
        Dates that should be generated.
    write_to_azure
        If True, append generated rows to Azure ``holdings`` table.
    configs_path
        Optional config path used when ``write_to_azure`` is True.

    Returns
    -------
    pandas.DataFrame
        Backfilled holdings rows formatted to match the holdings table schema.
    """
    dates = _normalize_target_dates(target_dates)
    source_ts = pd.Timestamp(source_date).normalize()
    log(f"Building holdings backfill for index='{index_name}' and {len(dates)} dates.")

    holdings = get_index_holdings(indices=index_name, start_date=source_ts)
    if holdings.empty:
        raise ValueError(
            f"No holdings found for index '{index_name}' from source_date {source_ts.date()}."
        )

    holdings = holdings.copy()
    holdings["DATE"] = pd.to_datetime(holdings["DATE"]).dt.normalize()
    base_snapshot = holdings[holdings["DATE"] == source_ts].copy()
    if base_snapshot.empty:
        available_dates = sorted(holdings["DATE"].dt.date.unique().tolist())
        raise ValueError(
            f"No holdings found for index '{index_name}' on source_date {source_ts.date()}. "
            f"Available dates from query: {available_dates[:10]}"
        )
    log(
        f"Using source snapshot date {source_ts.date()} with {len(base_snapshot)} rows."
    )

    all_columns = base_snapshot.columns.tolist()
    base_meta = base_snapshot.drop(["DATE", "WEIGHT", "MARKET_VALUE", "PRICE"], axis=1)

    base_weights = base_snapshot.pivot(
        index="DATE", columns="TICKER", values="WEIGHT"
    ).reset_index()
    weight_schedule = _build_weight_schedule(
        base_weights=base_weights, target_dates=dates
    )
    in_index = weight_schedule > 0

    price_start_date = min(source_ts, dates.min()).date().isoformat()
    prices = get_prices(weight_schedule.columns, start_date=price_start_date)
    prices = prices.reindex(weight_schedule.index).ffill()

    if prices.empty:
        raise ValueError("No price data available for the selected tickers/dates.")

    # Drop tickers with no usable prices across all requested dates.
    missing_price_tickers = prices.columns[prices.isna().all()].tolist()
    if missing_price_tickers:
        log(
            f"Dropping {len(missing_price_tickers)} tickers with no prices: "
            f"{', '.join(missing_price_tickers)}",
            type="warning",
        )
        prices = prices.drop(columns=missing_price_tickers)
        weight_schedule = weight_schedule.drop(
            columns=missing_price_tickers, errors="ignore"
        )
        in_index = in_index.drop(columns=missing_price_tickers, errors="ignore")

    strategy = Strategy(
        name="buy_and_hold",
        algos=[
            RunOnce(),
            SelectWhere(in_index),
            SelectAll(),
            SelectActive(),
            WeighTarget(weights=weight_schedule),
            Rebalance(),
        ],
    )
    result = bt.run(Backtest(strategy=strategy, data=prices, integer_positions=False))

    prices_long = prices.melt(
        ignore_index=False, var_name="TICKER", value_name="PRICE"
    ).reset_index(names="DATE")
    prices_long["PRICE"] = np.round(prices_long["PRICE"], 2)

    weights_long = (
        result.backtests["buy_and_hold"]
        .security_weights.loc[dates]
        .melt(ignore_index=False, var_name="TICKER", value_name="WEIGHT")
        .reset_index(names="DATE")
    )

    out = weights_long.merge(base_meta, how="left", on="TICKER")
    out = out.merge(prices_long, how="left", on=["DATE", "TICKER"])
    out["MARKET_VALUE"] = out["QUANTITY"] * out["PRICE"]
    out = out[out["WEIGHT"] > 0].copy()
    out = out[all_columns]
    out = out.sort_values(["DATE", "TICKER"]).reset_index(drop=True)

    if write_to_azure:
        engine = default_azure_data_source.get_engine(configs_path=configs_path)
        default_azure_data_source.write_sql_table(
            table_name="holdings",
            engine=engine,
            df=out,
            overwrite=False,
            dtype_overrides={"DATE": satypes.Date()},
        )
        log(f"Wrote {len(out)} backfill rows to Azure table 'holdings'.")

    log(f"Backfill completed for index='{index_name}': {len(out)} rows.")
    return out


if __name__ == "__main__":
    demo_dates = ["2026-02-20", "2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26"]
    dataframe = build_index_holdings_backfill(
        index_name="Russell 1000",
        source_date="2026-02-27",
        target_dates=demo_dates,
        write_to_azure=True,
    )
    print(dataframe.head())
