from unittest.mock import MagicMock

import pandas as pd

from pipelines.daily_market_data.insider_transactions_data import (
    InsiderTransactionsData,
)


def test_insider_transactions_run_pulls_with_retry_helper():
    pipeline = InsiderTransactionsData(tickers=["AAPL", "MSFT"])
    pulled = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)

    out = pipeline.run()

    pipeline._pull_with_missing_ticker_retries.assert_called_once()
    kwargs = pipeline._pull_with_missing_ticker_retries.call_args.kwargs
    assert kwargs["client_method"] == "get_insider_transactions"
    assert kwargs["max_resets"] == 10
    assert kwargs["wait_seconds"] == 120
    assert len(out) == 1
    assert out.iloc[0]["TICKER"] == "AAPL"
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_insider_transactions_run_coerces_date_columns():
    pipeline = InsiderTransactionsData(tickers=["AAPL"])
    pulled = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "START_DATE": ["2023-12-31"],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)

    out = pipeline.run()

    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["START_DATE"])


def test_insider_transactions_run_default_does_not_write_to_azure():
    pipeline = InsiderTransactionsData(tickers=["AAPL"])
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run()

    pipeline.azure_data_source.write_sql_table.assert_not_called()


def test_insider_transactions_run_write_to_azure_writes_table():
    pipeline = InsiderTransactionsData(tickers=["AAPL"])
    pulled = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02T14:30:00Z"],
            "START_DATE": ["2023-12-31T01:05:00Z"],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    pipeline.azure_data_source.get_engine.assert_called_once_with(
        configs_path="config/configs.json"
    )
    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "insider_transactions"
    assert kwargs["overwrite"] is False
    assert set(kwargs["dtype_overrides"]) == {"DATE", "START_DATE"}
    assert str(kwargs["df"]["DATE"].iloc[0]) == "2024-01-02"
    assert str(kwargs["df"]["START_DATE"].iloc[0]) == "2023-12-31"


def test_insider_transactions_run_write_to_azure_skips_rows_duplicated_minus_date():
    pipeline = InsiderTransactionsData(tickers=["AAPL"])
    pulled = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "INSIDER": ["John Doe"],
            "SHARES": [1000],
            "START_DATE": ["2023-12-31"],
        }
    )
    existing = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-01"],
            "INSIDER": ["John Doe"],
            "SHARES": [1000],
            "START_DATE": [pd.Timestamp("2023-12-31")],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=existing)
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True)

    pipeline.azure_data_source.write_sql_table.assert_not_called()
