"""Daily market data pipeline runner."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure repository-root imports work when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.daily_market_data import (
    AnalystPriceTargetsData,
    AnalystRecommendationsData,
    AnalystUpgradesDowngradesData,
    DownloadHoldings,
    EPSRevisionsData,
    EstimatesData,
    ETF_URLS,
    FinancialData,
    InfoData,
    InsiderTransactionsData,
    InstitutionalHolders,
    MajorHolders,
    OptionsData,
    PricingData,
)
from handyman.holdings import get_index_holdings
from handyman.holdings import get_llm_holdings
from utils.logging import configure_file_logging, log

DEFAULT_INDICES = [
    "^GSPC",
    "^NDX",
    "^DJI",
    "^RUT",
    "^MID",
    "^RUI",
    "VEA",
    "^FTSE",
    "^GDAXI",
    "^N225",
    "^FCHI",
    "^STOXX50E",
    "EEM",
    "VWO",
    "^HSI",
    "000001.SS",
    "^BSESN",
    "^BVSP",
    "^TNX",
    "^IRX",
    "^TYX",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "BND",
    "^VIX",
    "^VVIX",
    "GC=F",
    "SI=F",
    "CL=F",
    "HG=F",
    "NG=F",
    "DBC",
    "DX-Y.NYB",
    "EURUSD=X",
    "JPY=X",
    "GBPUSD=X",
    "CNY=X",
    "VNQ",
    "IYR",
    "MTUM",
    "VLUE",
    "QUAL",
    "SIZE",
    "USMV",
    "BTC-USD",
    "ETH-USD",
]


def run_daily_market_data(
    *,
    write_to_db: bool = False,
    configs_path: str | None = None,
    indices: list[str] | None = None,
    include_options: bool = False,
    use_latest_holdings_snapshot: bool = False,
) -> dict[str, object]:
    """Run the full daily market data workflow and return produced datasets."""
    selected_indices = indices if indices is not None else DEFAULT_INDICES
    log(
        "Starting daily market data pipeline: "
        f"{len(ETF_URLS)} funds, {len(selected_indices)} index tickers, "
        f"write_to_db={write_to_db}"
    )

    if use_latest_holdings_snapshot:
        log("Loading holdings from latest database snapshot (rerun shortcut).")
        df_holdings = get_index_holdings(
            get_latest=True,
            configs_path=configs_path,
        )
        if df_holdings.empty:
            log(
                "Latest holdings snapshot is empty; falling back to DownloadHoldings pull.",
                type="warning",
            )
            holdings_frames = [
                DownloadHoldings(fund_name=fund, url=url).run(
                    write_to_azure=write_to_db,
                    configs_path=configs_path,
                )
                for fund, url in ETF_URLS.items()
            ]
            df_holdings = pd.concat(holdings_frames, ignore_index=True)
    else:
        log("Running holdings pipeline.")
        holdings_frames = [
            DownloadHoldings(fund_name=fund, url=url).run(
                write_to_azure=write_to_db,
                configs_path=configs_path,
            )
            for fund, url in ETF_URLS.items()
        ]
        df_holdings = pd.concat(holdings_frames, ignore_index=True)
    holdings_tickers = df_holdings["TICKER"].dropna().astype(str).unique()
    log(f"Holdings pipeline complete: {len(df_holdings)} rows")

    log("Running index pricing pipeline.")
    index_prices = PricingData(tickers=selected_indices).run(
        write_to_azure=write_to_db,
        use_start_date_mapping=True,
        configs_path=configs_path,
    )
    log(f"Index pricing pipeline complete: {len(index_prices)} rows")

    log("Running holdings pricing pipeline.")
    all_prices = PricingData(tickers=holdings_tickers).run(
        write_to_azure=write_to_db,
        use_start_date_mapping=True,
        adjust_for_corporate_actions=True,
        configs_path=configs_path,
    )
    log(f"Holdings pricing pipeline complete: {len(all_prices)} rows")

    log("Running analyst price targets pipeline.")
    all_analyst_price_targets = AnalystPriceTargetsData(tickers=holdings_tickers).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
    )
    log(
        f"Analyst price targets pipeline complete: {len(all_analyst_price_targets)} rows"
    )

    log("Running company info/officers pipeline.")
    all_info, all_officers = InfoData(tickers=holdings_tickers).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
    )
    log(
        f"Info pipeline complete: info={len(all_info)} rows, officers={len(all_officers)} rows"
    )

    if include_options:
        log("Running options pipeline.")
        all_options = OptionsData(tickers=holdings_tickers).run(
            write_to_azure=write_to_db,
            configs_path=configs_path,
        )
        log(f"Options pipeline complete: {len(all_options)} rows")
    else:
        all_options = pd.DataFrame()
        log("Skipping options pipeline (include_options=False).")

    log("Running analyst recommendations pipeline.")
    all_analyst_recommendations = AnalystRecommendationsData(
        tickers=holdings_tickers
    ).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
    )
    all_analyst_upgrades_downgrades = AnalystUpgradesDowngradesData(
        tickers=holdings_tickers
    ).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
    )
    log(
        "Analyst recommendations pipeline complete: "
        f"recommendations={len(all_analyst_recommendations)} rows, "
        f"upgrades_downgrades={len(all_analyst_upgrades_downgrades)} rows"
    )

    log("Running holders pipeline.")
    holder_tickers = holdings_tickers
    all_institutional_holders = InstitutionalHolders(tickers=holder_tickers).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
    )
    all_major_holders = MajorHolders(tickers=holder_tickers).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
    )
    log(
        "Holders pipeline complete: "
        f"institutional={len(all_institutional_holders)} rows, "
        f"major={len(all_major_holders)} rows"
    )

    log("Running insider transactions pipeline.")
    all_insider_transactions = InsiderTransactionsData(tickers=holdings_tickers).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
    )
    log(f"Insider transactions pipeline complete: {len(all_insider_transactions)} rows")

    log("Running financial statements pipeline.")
    financial_data: dict[str, pd.DataFrame] = FinancialData(
        tickers=holdings_tickers,
    ).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
    )

    log("Running EPS revisions pipeline.")
    all_eps_revisions = EPSRevisionsData(
        tickers=holdings_tickers,
    ).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
    )
    log(f"EPS revisions pipeline complete: {len(all_eps_revisions)} rows")

    log("Running analyst estimates pipeline.")
    estimates_data: dict[str, pd.DataFrame] = EstimatesData(
        tickers=holdings_tickers,
    ).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
    )
    all_eps_estimates = estimates_data["eps"]
    all_revenue_estimates = estimates_data["revenue"]
    all_growth_estimates = estimates_data["growth"]
    log(
        "Analyst estimates pipeline complete: "
        f"eps={len(all_eps_estimates)} rows, "
        f"revenue={len(all_revenue_estimates)} rows, "
        f"growth={len(all_growth_estimates)} rows"
    )

    log("Daily market data pipeline finished.")
    output: dict[str, object] = {
        "tickers_run": {
            "holdings_tickers": list(pd.Index(holdings_tickers).astype(str)),
            "index_tickers": list(pd.Index(selected_indices).astype(str)),
        },
        "holdings": df_holdings,
        "index_prices": index_prices,
        "all_prices": all_prices,
        "all_analyst_price_targets": all_analyst_price_targets,
        "all_info": all_info,
        "all_officers": all_officers,
        "all_options": all_options,
        "all_analyst_recommendations": all_analyst_recommendations,
        "all_analyst_upgrades_downgrades": all_analyst_upgrades_downgrades,
        "all_institutional_holders": all_institutional_holders,
        "all_major_holders": all_major_holders,
        "all_insider_transactions": all_insider_transactions,
        "all_eps_revisions": all_eps_revisions,
        "all_eps_estimates": all_eps_estimates,
        "all_revenue_estimates": all_revenue_estimates,
        "all_growth_estimates": all_growth_estimates,
        "financial_data": financial_data,
    }
    return output


def run_market_data_for_tickers(
    *,
    tickers: list[str],
    write_to_db: bool = False,
    configs_path: str | None = None,
) -> dict[str, object]:
    """Run a lightweight market data workflow for an explicit ticker list.

    This run executes only:
    - prices (PricingData)
    - company info/officers (InfoData)
    """
    ticker_list = list(pd.Index(tickers).dropna().astype(str).str.strip())
    ticker_list = [t for t in ticker_list if t]
    if not ticker_list:
        raise ValueError("tickers must contain at least one non-empty ticker.")

    log(
        "Starting ticker-only market data pipeline: "
        f"{len(ticker_list)} tickers, write_to_db={write_to_db}"
    )

    log("Running pricing pipeline.")
    all_prices = PricingData(tickers=ticker_list).run(
        write_to_azure=write_to_db,
        use_start_date_mapping=True,
        adjust_for_corporate_actions=True,
        configs_path=configs_path,
    )
    log(f"Pricing pipeline complete: {len(all_prices)} rows")

    log("Running company info/officers pipeline.")
    all_info, all_officers = InfoData(tickers=ticker_list).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
    )
    log(
        f"Info pipeline complete: info={len(all_info)} rows, officers={len(all_officers)} rows"
    )

    log("Ticker-only market data pipeline finished.")
    return {
        "tickers_run": ticker_list,
        "all_prices": all_prices,
        "all_info": all_info,
        "all_officers": all_officers,
    }


if __name__ == "__main__":
    # tail -f "$(ls -t logs/daily_market_data_*.log | head -n 1)"
    write_to_db = True
    configs_path = None
    include_options = False
    use_latest_holdings_snapshot = False
    log_dir = PROJECT_ROOT / "logs"
    log_file = log_dir / f"daily_market_data_{datetime.now():%Y%m%d_%H%M%S}.log"
    configure_file_logging(log_file=str(log_file))
    log(f"File logging enabled: {log_file}", type="info")

    daily_output = run_daily_market_data(
        write_to_db=write_to_db,
        include_options=include_options,
        configs_path=configs_path,
        use_latest_holdings_snapshot=use_latest_holdings_snapshot,
    )
    ran_tickers = set(
        pd.Index(daily_output["tickers_run"]["holdings_tickers"])
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    latest_llm_holdings = get_llm_holdings(get_latest=True, configs_path=configs_path)
    llm_tickers = set(
        pd.Index(latest_llm_holdings.get("TICKER", pd.Series(dtype=str)))
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    missing_tickers = sorted(t for t in llm_tickers if t and t not in ran_tickers)

    if missing_tickers:
        log(
            f"Running ticker-only pipeline for {len(missing_tickers)} missing LLM tickers.",
            type="info",
        )
        run_market_data_for_tickers(
            tickers=missing_tickers,
            write_to_db=write_to_db,
            configs_path=configs_path,
        )
    else:
        log("No missing LLM tickers to run after daily market pipeline.", type="info")
