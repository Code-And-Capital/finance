"""Backfill index holdings over a date range via holdings + price replay."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# Ensure repository-root imports work when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bt
from bt.algos.flow import ClosePositionsAfterDates, RunAfterDate, RunOnce
from bt.algos.portfolio_ops import Rebalance
from bt.algos.selection import SelectAll, SelectWhere, SelectActive
from bt.algos.weighting import WeightFixedSchedule
from bt.core import Strategy
from bt.engine import Backtest
from sqlalchemy import types as satypes

from connectors.azure_data_source import default_azure_data_source
from data_loading.runner import Runner
from utils.logging import log


def build_index_holdings_backfill(
    *,
    index_name: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    write_to_azure: bool = False,
    configs_path: str | None = None,
) -> pd.DataFrame:
    """Build index holdings from replayed weights over a date range.

    Parameters
    ----------
    index_name
        Index name as stored in holdings table (for example ``"S&P 500"``).
    start_date
        First date (inclusive) for holdings replay input. This should be the
        last date that still has index holdings available in the source table.
    end_date
        Last date (inclusive) for holdings replay input.
    write_to_azure
        If True, append generated rows to Azure ``holdings`` table.
    configs_path
        Optional config path used when ``write_to_azure`` is True.

    Returns
    -------
    pandas.DataFrame
        Backfilled holdings rows formatted to match the holdings table schema.
    """
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if end_ts < start_ts:
        raise ValueError("end_date must be on or after start_date.")

    log(
        "Building holdings backfill for "
        f"index='{index_name}' start_date={start_ts.date()} end_date={end_ts.date()}",
        type="info",
    )

    log("Running data-loading Runner for holdings/prices context.", type="info")
    runner = Runner(
        portfolio=[index_name],
        start_date=start_ts.date().isoformat(),
        end_date=end_ts.date().isoformat(),
        configs_path=configs_path,
    )
    _ = runner.run()

    weights_wide = runner.holdings_data_source.formatted_data["weights_wide"]
    in_portfolio = runner.holdings_data_source.formatted_data["in_portfolio_wide"]
    prices = runner.prices_data_source.formatted_data["prices_wide_full_history"].shift(
        1
    )
    last_valid_date = runner.prices_data_source.formatted_data["last_valid_date"]

    if weights_wide.empty or in_portfolio.empty or prices.empty:
        raise ValueError("Runner outputs are empty; cannot build backfill holdings.")

    prior_dates = prices.index[prices.index < start_ts]
    output_dates = prices.index[prices.index > start_ts]
    if len(prior_dates) == 0 or len(output_dates) == 0:
        raise ValueError(
            "Insufficient price history around holdings start; cannot derive run/output dates."
        )
    start_date_run = prior_dates[-1]
    start_date_output = output_dates[0]

    log(
        "Derived replay dates "
        f"start_date_run={pd.Timestamp(start_date_run).date()} "
        f"start_date_output={pd.Timestamp(start_date_output).date()}",
        type="info",
    )

    strategy = Strategy(
        name="buy_and_hold",
        algos=[
            ClosePositionsAfterDates(close_dates=last_valid_date),
            RunAfterDate(start_date_run),
            RunOnce(),
            SelectWhere(in_portfolio),
            SelectAll(),
            SelectActive(),
            WeightFixedSchedule(weights=weights_wide),
            Rebalance(),
        ],
    )
    log("Running backtest replay for holdings backfill.", type="info")
    result = bt.run(Backtest(strategy=strategy, data=prices, integer_positions=False))

    prices_long = prices.melt(
        ignore_index=False, var_name="FIGI", value_name="PRICE"
    ).reset_index(names="DATE")
    prices_long["PRICE"] = np.round(prices_long["PRICE"], 2)

    weights_long = (
        result.backtests["buy_and_hold"]
        .security_weights.loc[start_date_output:]
        .melt(ignore_index=False, var_name="FIGI", value_name="WEIGHT")
        .reset_index(names="DATE")
    )

    holdings = runner.holdings_data_source.formatted_data["holdings_long"].copy()
    holdings["DATE"] = pd.to_datetime(holdings["DATE"]).dt.normalize()
    holdings = holdings[holdings["DATE"] == start_ts]
    all_columns = holdings.columns.tolist()
    base_meta = holdings.drop(["DATE", "WEIGHT", "MARKET_VALUE", "PRICE"], axis=1)

    out = weights_long.merge(base_meta, how="left", on="FIGI")
    out = out.merge(prices_long, how="left", on=["DATE", "FIGI"])
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
        log(f"Wrote {len(out)} backfill rows to Azure table 'holdings'.", type="info")

    log(f"Backfill completed for index='{index_name}': {len(out)} rows.", type="info")
    return out


if __name__ == "__main__":
    params: dict[str, Any] = {
        "index_name": "Russell 1000",
        "start_date": "2026-02-20",
        "end_date": "2026-02-27",
        "write_to_azure": True,
    }
    dataframe = build_index_holdings_backfill(
        **params,
    )
    print(dataframe.head())
