from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from pipelines.daily_market_data.index_holdings import DownloadHoldings


def test_download_holdings_run_full_processing(monkeypatch):
    downloader = DownloadHoldings(
        fund_name="TESTFUND",
        url="http://dummy",
        download_folder="/tmp",
    )

    mock_download = MagicMock(return_value=None)
    monkeypatch.setattr(
        downloader.selenium_data_source, "download_data_file", mock_download
    )

    fake_raw_df = pd.DataFrame(
        {
            "ASSET_CLASS": ["Equity", "Equity", "Bond", "Equity"],
            "TICKER": ["AAPL", "--", "MSFT", "BAD"],
            "NAME": ["Apple", "BadRow", "Microsoft", "BadCorp"],
            "MARKET_VALUE": ["100", "999", "200", "300"],
            "WEIGHT": ["0.4", "0.1", "-0.2", "0.6"],
            "QUANTITY": ["10", "999", "5", "20"],
            "PRICE": ["10", "1", "40", "15"],
            "LOCATION": ["USA", "USA", "USA", "USA"],
            "EXCHANGE": [
                "NASDAQ",
                "New York Stock Exchange Inc.",
                "Nyse Mkt Llc",
                "InvalidExchange",
            ],
            "CURRENCY": ["USD", "USD", "USD", "USD"],
            "FX_RATE": ["1.0", "1.0", "1.0", "1.0"],
        }
    )
    monkeypatch.setattr(
        downloader.xls_data_source,
        "read_xls_file",
        MagicMock(return_value=fake_raw_df),
    )

    import pipelines.daily_market_data.index_holdings as module

    monkeypatch.setattr(module, "ETF_FILE_NAMES", {"TESTFUND": "fakefile.xlsx"})
    monkeypatch.setattr(module, "TICKER_MAPPING", {"BAD": "GOOD"})
    monkeypatch.setattr(module.os.path, "exists", lambda _: True)
    monkeypatch.setattr(
        downloader,
        "_run_openfigi_pipeline",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        downloader,
        "_load_security_master",
        lambda **_kwargs: pd.DataFrame(
            {"TICKER": ["AAPL"], "NAME": ["Apple"], "FIGI": ["FIGI_AAPL"]}
        ),
    )

    removed_paths: list[str] = []
    monkeypatch.setattr(module.os, "remove", lambda p: removed_paths.append(p))

    result = downloader.run()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["TICKER"] == "AAPL"
    assert row["EXCHANGE"] == "NASDAQ"
    assert abs(row["WEIGHT"] - 1.0) < 1e-6
    assert row["INDEX"] == "TESTFUND"
    assert row["DATE"] == date.today()
    assert removed_paths == [f"{downloader.download_folder}/fakefile.xlsx"]
    mock_download.assert_called_once_with(
        url="http://dummy",
        download_folder=downloader.download_folder,
    )


def test_resolve_file_path_unknown_fund_raises():
    downloader = DownloadHoldings(fund_name="UNKNOWN", url="http://dummy")
    with pytest.raises(KeyError, match="Unknown fund_name"):
        downloader._resolve_file_path()


def test_validate_row_count_sp500_in_range_passes():
    downloader = DownloadHoldings(
        fund_name="S&P 500",
        url="http://dummy",
        download_folder="/tmp",
    )
    df = pd.DataFrame({"TICKER": [f"T{i}" for i in range(500)]})
    downloader._validate_row_count(df)


def test_validate_row_count_sp500_out_of_range_raises():
    downloader = DownloadHoldings(
        fund_name="S&P 500",
        url="http://dummy",
        download_folder="/tmp",
    )
    df = pd.DataFrame({"TICKER": [f"T{i}" for i in range(300)]})
    with pytest.raises(ValueError, match="row-count check failed"):
        downloader._validate_row_count(df)


def test_run_default_does_not_write_to_azure(monkeypatch):
    downloader = DownloadHoldings(
        fund_name="TESTFUND",
        url="http://dummy",
        download_folder="/tmp",
    )

    import pipelines.daily_market_data.index_holdings as module

    monkeypatch.setattr(module, "ETF_FILE_NAMES", {"TESTFUND": "fakefile.xlsx"})
    monkeypatch.setattr(
        downloader.selenium_data_source, "download_data_file", lambda **_: None
    )
    monkeypatch.setattr(module.os.path, "exists", lambda _: True)
    monkeypatch.setattr(module.os, "remove", lambda _: None)
    monkeypatch.setattr(
        downloader,
        "_run_openfigi_pipeline",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        downloader,
        "_load_security_master",
        lambda **_kwargs: pd.DataFrame(
            {"TICKER": ["AAPL"], "NAME": ["Apple"], "FIGI": ["FIGI_AAPL"]}
        ),
    )

    fake_raw_df = pd.DataFrame(
        {
            "ASSET_CLASS": ["Equity"],
            "TICKER": ["AAPL"],
            "NAME": ["Apple"],
            "MARKET_VALUE": ["100"],
            "WEIGHT": ["1.0"],
            "QUANTITY": ["10"],
            "PRICE": ["10"],
            "LOCATION": ["USA"],
            "EXCHANGE": ["NASDAQ"],
            "CURRENCY": ["USD"],
            "FX_RATE": ["1.0"],
        }
    )
    monkeypatch.setattr(
        downloader.xls_data_source, "read_xls_file", lambda **_: fake_raw_df
    )

    mock_write = MagicMock(return_value=None)
    monkeypatch.setattr(downloader.azure_data_source, "write_sql_table", mock_write)

    downloader.run()
    mock_write.assert_not_called()


def test_run_write_to_azure_writes_holdings(monkeypatch):
    downloader = DownloadHoldings(
        fund_name="TESTFUND",
        url="http://dummy",
        download_folder="/tmp",
    )

    import pipelines.daily_market_data.index_holdings as module

    monkeypatch.setattr(module, "ETF_FILE_NAMES", {"TESTFUND": "fakefile.xlsx"})
    monkeypatch.setattr(
        downloader.selenium_data_source, "download_data_file", lambda **_: None
    )
    monkeypatch.setattr(module.os.path, "exists", lambda _: True)
    monkeypatch.setattr(module.os, "remove", lambda _: None)
    monkeypatch.setattr(
        downloader,
        "_run_openfigi_pipeline",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        downloader,
        "_load_security_master",
        lambda **_kwargs: pd.DataFrame(
            {"TICKER": ["AAPL"], "NAME": ["Apple"], "FIGI": ["FIGI_AAPL"]}
        ),
    )

    fake_raw_df = pd.DataFrame(
        {
            "ASSET_CLASS": ["Equity"],
            "TICKER": ["AAPL"],
            "NAME": ["Apple"],
            "MARKET_VALUE": ["100"],
            "WEIGHT": ["1.0"],
            "QUANTITY": ["10"],
            "PRICE": ["10"],
            "LOCATION": ["USA"],
            "EXCHANGE": ["NASDAQ"],
            "CURRENCY": ["USD"],
            "FX_RATE": ["1.0"],
        }
    )
    monkeypatch.setattr(
        downloader.xls_data_source, "read_xls_file", lambda **_: fake_raw_df
    )

    mock_get_engine = MagicMock(return_value=object())
    mock_write = MagicMock(return_value=None)
    monkeypatch.setattr(downloader.azure_data_source, "get_engine", mock_get_engine)
    monkeypatch.setattr(downloader.azure_data_source, "write_sql_table", mock_write)

    result = downloader.run(write_to_azure=True, configs_path="config/configs.json")

    mock_get_engine.assert_called_once_with(configs_path="config/configs.json")
    mock_write.assert_called_once()
    kwargs = mock_write.call_args.kwargs
    assert kwargs["table_name"] == "holdings"
    assert kwargs["overwrite"] is False
    assert kwargs["df"].equals(result)


def test_delete_rows_by_where_clause(monkeypatch):
    downloader = DownloadHoldings(
        fund_name="TESTFUND",
        url="http://dummy",
        download_folder="/tmp",
    )

    mock_get_engine = MagicMock(return_value=object())
    mock_delete = MagicMock(return_value=None)
    mock_build_delete = MagicMock(
        return_value="DELETE FROM [dbo].[holdings] WHERE DATE = '2025-12-29'"
    )
    monkeypatch.setattr(downloader.azure_data_source, "get_engine", mock_get_engine)
    monkeypatch.setattr(downloader.azure_data_source, "delete_sql_rows", mock_delete)
    monkeypatch.setattr(downloader.sql_client, "build_delete_query", mock_build_delete)

    downloader.delete_rows_by_where_clause(
        where_clause="DATE = '2025-12-29'",
        configs_path="config/configs.json",
    )

    mock_build_delete.assert_called_once_with(
        table_name="holdings",
        where_clause="DATE = '2025-12-29'",
        schema="dbo",
    )
    mock_delete.assert_called_once()
    delete_kwargs = mock_delete.call_args.kwargs
    assert delete_kwargs["query"].startswith("DELETE FROM")


def test_run_removes_downloaded_file_when_load_fails(monkeypatch):
    downloader = DownloadHoldings(
        fund_name="TESTFUND",
        url="http://dummy",
        download_folder="/tmp",
    )

    import pipelines.daily_market_data.index_holdings as module

    monkeypatch.setattr(module, "ETF_FILE_NAMES", {"TESTFUND": "fakefile.xlsx"})
    monkeypatch.setattr(
        downloader.selenium_data_source, "download_data_file", lambda **_: None
    )
    monkeypatch.setattr(module.os.path, "exists", lambda _: True)

    removed_paths: list[str] = []
    monkeypatch.setattr(module.os, "remove", lambda p: removed_paths.append(p))
    monkeypatch.setattr(
        downloader.xls_data_source,
        "read_xls_file",
        MagicMock(side_effect=RuntimeError("bad xls")),
    )

    with pytest.raises(RuntimeError, match="bad xls"):
        downloader.run()

    assert removed_paths == [f"{downloader.download_folder}/fakefile.xlsx"]


def test_attach_figi_merges_on_ticker_and_name_and_warns_missing(monkeypatch):
    downloader = DownloadHoldings(
        fund_name="TESTFUND",
        url="http://dummy",
        download_folder="/tmp",
    )
    holdings = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "NAME": ["Apple", "Microsoft"],
            "WEIGHT": [0.5, 0.5],
        }
    )
    security_master = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "NAME": ["Apple"],
            "FIGI": ["FIGI_AAPL"],
        }
    )

    import pipelines.daily_market_data.index_holdings as module

    warned: list[str] = []
    monkeypatch.setattr(
        module,
        "log",
        lambda msg, type=None: warned.append(msg) if type == "warning" else None,
    )

    out = downloader._attach_figi(
        holdings_df=holdings,
        security_master_override=security_master,
    )

    assert out.loc[out["TICKER"] == "AAPL", "FIGI"].iloc[0] == "FIGI_AAPL"
    assert out.loc[out["TICKER"] == "MSFT", "FIGI"].isna().iloc[0]
    assert any("Holdings tickers missing from security_master" in m for m in warned)


def test_dedupe_security_master_prefers_united_states_location():
    downloader = DownloadHoldings(
        fund_name="TESTFUND",
        url="http://dummy",
        download_folder="/tmp",
    )
    security_master = pd.DataFrame(
        {
            "TICKER": ["ABC", "ABC"],
            "NAME": ["Alpha", "Alpha"],
            "LOCATION": ["Canada", "United States"],
            "DATE": ["2026-03-01", "2026-03-02"],
            "FIGI": ["FIGI_CA", "FIGI_US"],
        }
    )

    deduped = downloader._dedupe_security_master_for_holdings(security_master)
    assert len(deduped) == 1
    assert deduped["FIGI"].iloc[0] == "FIGI_US"


def test_dedupe_security_master_prefers_earliest_date_when_no_united_states():
    downloader = DownloadHoldings(
        fund_name="TESTFUND",
        url="http://dummy",
        download_folder="/tmp",
    )
    security_master = pd.DataFrame(
        {
            "TICKER": ["XYZ", "XYZ"],
            "NAME": ["Xylon", "Xylon"],
            "LOCATION": ["Canada", "Canada"],
            "DATE": ["2026-03-05", "2026-03-01"],
            "FIGI": ["FIGI_NEWER", "FIGI_OLDER"],
        }
    )

    deduped = downloader._dedupe_security_master_for_holdings(security_master)
    assert len(deduped) == 1
    assert deduped["FIGI"].iloc[0] == "FIGI_OLDER"
