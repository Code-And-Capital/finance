from unittest.mock import MagicMock

import pandas as pd

from pipelines.daily_market_data.holders_data import InstitutionalHolders, MajorHolders


def test_institutional_holders_run_pulls_with_retry_helper():
    pipeline = InstitutionalHolders(tickers=["AAPL"])
    pulled = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "DATE_REPORTED": ["2023-12-31"],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)

    out = pipeline.run()

    pipeline._pull_with_missing_ticker_retries.assert_called_once()
    kwargs = pipeline._pull_with_missing_ticker_retries.call_args.kwargs
    assert kwargs["client_method"] == "get_institutional_holders"
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["DATE_REPORTED"])


def test_major_holders_run_pulls_with_retry_helper():
    pipeline = MajorHolders(tickers=["AAPL"])
    pulled = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)

    out = pipeline.run()

    pipeline._pull_with_missing_ticker_retries.assert_called_once()
    kwargs = pipeline._pull_with_missing_ticker_retries.call_args.kwargs
    assert kwargs["client_method"] == "get_major_holders"
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_institutional_holders_write_to_azure_writes_expected_table():
    pipeline = InstitutionalHolders(tickers=["AAPL"])
    pulled = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02T14:30:00Z"],
            "DATE_REPORTED": ["2023-12-31T01:05:00Z"],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "institutional_holders"
    assert set(kwargs["dtype_overrides"]) == {"DATE", "DATE_REPORTED"}


def test_major_holders_write_to_azure_writes_expected_table():
    pipeline = MajorHolders(tickers=["AAPL"])
    pulled = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02T10:10:00Z"]})
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "major_holders"
    assert set(kwargs["dtype_overrides"]) == {"DATE"}


def test_holders_skip_write_when_rows_duplicate_minus_date():
    institutional = InstitutionalHolders(tickers=["AAPL"])
    inst_incoming = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "HOLDER": ["Fund A"],
            "SHARES": [1000],
            "DATE_REPORTED": ["2023-12-31"],
        }
    )
    inst_existing = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "FIGI": ["FIGI_AAPL"],
            "DATE": ["2024-01-01"],
            "HOLDER": ["Fund A"],
            "SHARES": [1000],
            "DATE_REPORTED": [pd.Timestamp("2023-12-31")],
        }
    )
    institutional._pull_with_missing_ticker_retries = MagicMock(
        return_value=inst_incoming
    )
    institutional.azure_data_source.get_engine = MagicMock(return_value=object())
    institutional.azure_data_source.read_sql_table = MagicMock(
        return_value=inst_existing
    )
    institutional.azure_data_source.write_sql_table = MagicMock(return_value=None)
    institutional.run(write_to_azure=True, ticker_to_figi={"AAPL": "FIGI_AAPL"})
    institutional.azure_data_source.write_sql_table.assert_not_called()

    major = MajorHolders(tickers=["AAPL"])
    major_incoming = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "METRIC": ["insidersPercentHeld"],
            "VALUE": [0.2],
        }
    )
    major_existing = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "FIGI": ["FIGI_AAPL"],
            "DATE": ["2024-01-01"],
            "METRIC": ["insidersPercentHeld"],
            "VALUE": [0.2],
        }
    )
    major._pull_with_missing_ticker_retries = MagicMock(return_value=major_incoming)
    major.azure_data_source.get_engine = MagicMock(return_value=object())
    major.azure_data_source.read_sql_table = MagicMock(return_value=major_existing)
    major.azure_data_source.write_sql_table = MagicMock(return_value=None)
    major.run(write_to_azure=True, ticker_to_figi={"AAPL": "FIGI_AAPL"})
    major.azure_data_source.write_sql_table.assert_not_called()
