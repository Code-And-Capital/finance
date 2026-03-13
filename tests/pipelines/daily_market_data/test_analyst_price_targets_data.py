from unittest.mock import MagicMock

import pandas as pd

from pipelines.daily_market_data.pricing_data import AnalystPriceTargetsData


def test_analyst_price_targets_run_write_to_azure_writes_table():
    pipeline = AnalystPriceTargetsData(tickers=["AAPL"])
    pipeline._pull_with_missing_ticker_retries = MagicMock(
        return_value=pd.DataFrame(
            {"DATE": ["2024-01-02"], "TARGET_MEAN": [250.0], "TICKER": ["AAPL"]}
        )
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    out = pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    pipeline.azure_data_source.get_engine.assert_called_once_with(
        configs_path="config/configs.json"
    )
    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "analyst_price_targets"
    assert kwargs["overwrite"] is False
    assert set(kwargs["dtype_overrides"]) == {"DATE"}
    assert str(kwargs["df"]["DATE"].iloc[0]) == "2024-01-02"
    assert len(out) == 1


def test_analyst_price_targets_run_skips_duplicate_minus_date():
    pipeline = AnalystPriceTargetsData(tickers=["AAPL"])
    pipeline._pull_with_missing_ticker_retries = MagicMock(
        return_value=pd.DataFrame(
            {"DATE": ["2024-01-02"], "TARGET_MEAN": [250.0], "TICKER": ["AAPL"]}
        )
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(
        return_value=pd.DataFrame(
            {
                "DATE": ["2024-01-01"],
                "TARGET_MEAN": [250.0],
                "TICKER": ["AAPL"],
                "FIGI": ["FIGI_AAPL"],
            }
        )
    )
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    out = pipeline.run(write_to_azure=True, ticker_to_figi={"AAPL": "FIGI_AAPL"})

    pipeline.azure_data_source.write_sql_table.assert_not_called()
    assert out.empty
