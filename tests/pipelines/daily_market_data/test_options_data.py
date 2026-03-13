from unittest.mock import MagicMock

import pandas as pd

from pipelines.daily_market_data.options_data import OptionsData


def test_options_data_run_pulls_with_retry_helper():
    pipeline = OptionsData(tickers=["AAPL", "MSFT"])
    pulled = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)

    out = pipeline.run()

    pipeline._pull_with_missing_ticker_retries.assert_called_once()
    kwargs = pipeline._pull_with_missing_ticker_retries.call_args.kwargs
    assert kwargs["client_method"] == "get_options"
    assert kwargs["max_resets"] == 10
    assert kwargs["wait_seconds"] == 120
    assert len(out) == 1
    assert out.iloc[0]["TICKER"] == "AAPL"
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_options_data_run_coerces_date_columns():
    pipeline = OptionsData(tickers=["AAPL"])
    pulled = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "LASTTRADEDATE": ["2024-01-01"],
            "EXPIRATION": ["2024-12-20"],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)

    out = pipeline.run()

    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["LASTTRADEDATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["EXPIRATION"])


def test_options_data_run_default_does_not_write_to_azure():
    pipeline = OptionsData(tickers=["AAPL"])
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run()

    pipeline.azure_data_source.write_sql_table.assert_not_called()


def test_options_data_run_write_to_azure_writes_options_table():
    pipeline = OptionsData(tickers=["AAPL"])
    pulled = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "LASTTRADEDATE": ["2024-01-01T14:30:00Z"],
            "EXPIRATION": ["2024-12-20"],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    out = pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    pipeline.azure_data_source.get_engine.assert_called_once_with(
        configs_path="config/configs.json"
    )
    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "options"
    assert kwargs["overwrite"] is False
    assert set(kwargs["dtype_overrides"]) == {"DATE", "LASTTRADEDATE", "EXPIRATION"}
    assert str(kwargs["df"]["DATE"].iloc[0]) == "2024-01-02"
    assert str(kwargs["df"]["LASTTRADEDATE"].iloc[0]) == "2024-01-01"
    assert str(kwargs["df"]["EXPIRATION"].iloc[0]) == "2024-12-20"
    assert len(kwargs["df"]) == len(out)


def test_options_data_run_write_to_azure_skips_rows_duplicated_minus_date():
    pipeline = OptionsData(tickers=["AAPL"])
    pulled = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "CONTRACTSYMBOL": ["AAPL240119C00100000"],
            "OPTION_TYPE": ["Call"],
            "EXPIRATION": ["2024-01-19"],
        }
    )
    existing = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "FIGI": ["FIGI_AAPL"],
            "DATE": ["2024-01-01"],
            "CONTRACTSYMBOL": ["AAPL240119C00100000"],
            "OPTION_TYPE": ["Call"],
            "EXPIRATION": [pd.Timestamp("2024-01-19")],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=existing)
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    out = pipeline.run(write_to_azure=True, ticker_to_figi={"AAPL": "FIGI_AAPL"})

    pipeline.azure_data_source.write_sql_table.assert_not_called()
    assert out.empty
