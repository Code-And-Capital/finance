"""Daily market data pipeline runner."""

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

DEFAULT_INDICES: dict[str, str] = {
    "^GSPC": "BBG000000001",
    "^NDX": "BBG000000002",
    "^DJI": "BBG000000003",
    "^RUT": "BBG000000004",
    "^MID": "BBG000000005",
    "^RUI": "BBG000000006",
    "VEA": "BBG000000007",
    "^FTSE": "BBG000000008",
    "^GDAXI": "BBG000000009",
    "^N225": "BBG000000010",
    "^FCHI": "BBG000000011",
    "^STOXX50E": "BBG000000012",
    "EEM": "BBG000000013",
    "VWO": "BBG000000014",
    "^HSI": "BBG000000015",
    "000001.SS": "BBG000000016",
    "^BSESN": "BBG000000017",
    "^BVSP": "BBG000000018",
    "^TNX": "BBG000000019",
    "^IRX": "BBG000000020",
    "^TYX": "BBG000000021",
    "TLT": "BBG000000022",
    "IEF": "BBG000000023",
    "LQD": "BBG000000024",
    "HYG": "BBG000000025",
    "BND": "BBG000000026",
    "^VIX": "BBG000000027",
    "^VVIX": "BBG000000028",
    "GC=F": "BBG000000029",
    "SI=F": "BBG000000030",
    "CL=F": "BBG000000031",
    "HG=F": "BBG000000032",
    "NG=F": "BBG000000033",
    "DBC": "BBG000000034",
    "DX-Y.NYB": "BBG000000035",
    "EURUSD=X": "BBG000000036",
    "JPY=X": "BBG000000037",
    "GBPUSD=X": "BBG000000038",
    "CNY=X": "BBG000000039",
    "VNQ": "BBG000000040",
    "IYR": "BBG000000041",
    "MTUM": "BBG000000042",
    "VLUE": "BBG000000043",
    "QUAL": "BBG000000044",
    "SIZE": "BBG000000045",
    "USMV": "BBG000000046",
    "BTC-USD": "BBG000000047",
    "ETH-USD": "BBG000000048",
}


def _build_ticker_to_figi_map(df_holdings: pd.DataFrame) -> dict[str, str | None]:
    """Build ticker->FIGI mapping from holdings output."""
    if "TICKER" not in df_holdings.columns:
        return {}

    working = df_holdings.copy()
    working["TICKER"] = working["TICKER"].astype(str).str.strip().str.upper()
    if "FIGI" not in working.columns:
        working["FIGI"] = None

    mapping_df = (
        working[["TICKER", "FIGI"]]
        .dropna(subset=["TICKER"])
        .drop_duplicates(subset=["TICKER"], keep="first")
        .reset_index(drop=True)
    )
    return dict(zip(mapping_df["TICKER"], mapping_df["FIGI"]))


def run_daily_market_data(
    *,
    write_to_db: bool = False,
    configs_path: str | None = None,
    indices: dict[str, str] | None = None,
    include_options: bool = False,
    use_latest_holdings_snapshot: bool = False,
) -> dict[str, object]:
    """Run the full daily market data workflow and return produced datasets."""
    selected_indices = (
        {
            str(ticker).strip().upper(): str(figi).strip()
            for ticker, figi in indices.items()
        }
        if indices is not None
        else {
            str(ticker).strip().upper(): str(figi).strip()
            for ticker, figi in DEFAULT_INDICES.items()
        }
    )
    missing_index_figi = [
        ticker for ticker, figi in selected_indices.items() if not ticker or not figi
    ]
    if missing_index_figi:
        raise ValueError(
            "indices must be provided as ticker->FIGI mapping with non-empty FIGI "
            f"for every ticker. Missing FIGI for: {missing_index_figi}"
        )
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
    holdings_ticker_to_figi = _build_ticker_to_figi_map(df_holdings)
    holdings_tickers = list(holdings_ticker_to_figi.keys())
    mapped_holdings_count = sum(
        pd.notna(value) for value in holdings_ticker_to_figi.values()
    )
    log(
        f"Built holdings ticker->FIGI map: {len(holdings_ticker_to_figi)} tickers "
        f"({mapped_holdings_count} with FIGI)"
    )
    log(f"Holdings pipeline complete: {len(df_holdings)} rows")

    log("Running index pricing pipeline.")
    index_prices = PricingData(tickers=list(selected_indices.keys())).run(
        write_to_azure=write_to_db,
        use_start_date_mapping=True,
        configs_path=configs_path,
        ticker_to_figi=selected_indices,
    )
    log(f"Index pricing pipeline complete: {len(index_prices)} rows")

    log("Running holdings pricing pipeline.")
    all_prices = PricingData(tickers=holdings_tickers).run(
        write_to_azure=write_to_db,
        use_start_date_mapping=True,
        adjust_for_corporate_actions=True,
        configs_path=configs_path,
        ticker_to_figi=holdings_ticker_to_figi,
    )
    log(f"Holdings pricing pipeline complete: {len(all_prices)} rows")

    log("Running analyst price targets pipeline.")
    all_analyst_price_targets = AnalystPriceTargetsData(tickers=holdings_tickers).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
        ticker_to_figi=holdings_ticker_to_figi,
    )
    log(
        f"Analyst price targets pipeline complete: {len(all_analyst_price_targets)} rows"
    )

    log("Running company info/officers pipeline.")
    all_info, all_officers = InfoData(tickers=holdings_tickers).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
        ticker_to_figi=holdings_ticker_to_figi,
    )
    log(
        f"Info pipeline complete: info={len(all_info)} rows, officers={len(all_officers)} rows"
    )

    if include_options:
        log("Running options pipeline.")
        all_options = OptionsData(tickers=holdings_tickers).run(
            write_to_azure=write_to_db,
            configs_path=configs_path,
            ticker_to_figi=holdings_ticker_to_figi,
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
        ticker_to_figi=holdings_ticker_to_figi,
    )
    all_analyst_upgrades_downgrades = AnalystUpgradesDowngradesData(
        tickers=holdings_tickers
    ).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
        ticker_to_figi=holdings_ticker_to_figi,
    )
    log(
        "Analyst recommendations pipeline complete: "
        f"recommendations={len(all_analyst_recommendations)} rows, "
        f"upgrades_downgrades={len(all_analyst_upgrades_downgrades)} rows"
    )

    log("Running holders pipeline.")
    all_institutional_holders = InstitutionalHolders(tickers=holdings_tickers).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
        ticker_to_figi=holdings_ticker_to_figi,
    )
    all_major_holders = MajorHolders(tickers=holdings_tickers).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
        ticker_to_figi=holdings_ticker_to_figi,
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
        ticker_to_figi=holdings_ticker_to_figi,
    )
    log(f"Insider transactions pipeline complete: {len(all_insider_transactions)} rows")

    log("Running financial statements pipeline.")
    financial_data: dict[str, pd.DataFrame] = FinancialData(
        tickers=holdings_tickers,
    ).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
        ticker_to_figi=holdings_ticker_to_figi,
    )

    log("Running EPS revisions pipeline.")
    all_eps_revisions = EPSRevisionsData(
        tickers=holdings_tickers,
    ).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
        ticker_to_figi=holdings_ticker_to_figi,
    )
    log(f"EPS revisions pipeline complete: {len(all_eps_revisions)} rows")

    log("Running analyst estimates pipeline.")
    estimates_data: dict[str, pd.DataFrame] = EstimatesData(
        tickers=holdings_tickers,
    ).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
        ticker_to_figi=holdings_ticker_to_figi,
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
            "index_tickers": list(pd.Index(selected_indices.keys()).astype(str)),
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
    ticker_to_figi: dict[str, str | None] | None = None,
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
    normalized_ticker_to_figi = None
    if ticker_to_figi:
        normalized_ticker_to_figi = {
            str(ticker)
            .strip()
            .upper(): (
                str(figi).strip().upper()
                if pd.notna(figi) and str(figi).strip()
                else None
            )
            for ticker, figi in ticker_to_figi.items()
            if str(ticker).strip()
        }

    all_prices = PricingData(tickers=ticker_list).run(
        write_to_azure=write_to_db,
        use_start_date_mapping=True,
        adjust_for_corporate_actions=True,
        configs_path=configs_path,
        ticker_to_figi=normalized_ticker_to_figi,
    )
    log(f"Pricing pipeline complete: {len(all_prices)} rows")

    log("Running company info/officers pipeline.")
    all_info, all_officers = InfoData(tickers=ticker_list).run(
        write_to_azure=write_to_db,
        configs_path=configs_path,
        ticker_to_figi=normalized_ticker_to_figi,
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
    ran_ticker_to_figi = _build_ticker_to_figi_map(
        daily_output.get("holdings", pd.DataFrame())
    )
    ran_figis = {
        str(figi).strip().upper()
        for figi in ran_ticker_to_figi.values()
        if pd.notna(figi) and str(figi).strip()
    }
    latest_llm_holdings = get_llm_holdings(get_latest=True, configs_path=configs_path)
    llm_ticker_to_figi = _build_ticker_to_figi_map(latest_llm_holdings)
    llm_ticker_to_figi = {
        str(ticker)
        .strip()
        .upper(): (
            str(figi).strip().upper() if pd.notna(figi) and str(figi).strip() else None
        )
        for ticker, figi in llm_ticker_to_figi.items()
        if str(ticker).strip()
    }

    missing_tickers = sorted(
        ticker
        for ticker, figi in llm_ticker_to_figi.items()
        if ticker and figi and figi not in ran_figis and ticker not in ran_tickers
    )

    if missing_tickers:
        log(
            f"Running ticker-only pipeline for {len(missing_tickers)} missing LLM tickers.",
            type="info",
        )
        run_market_data_for_tickers(
            tickers=missing_tickers,
            ticker_to_figi=llm_ticker_to_figi,
            write_to_db=write_to_db,
            configs_path=configs_path,
        )
    else:
        log("No missing LLM tickers to run after daily market pipeline.", type="info")
