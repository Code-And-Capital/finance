from unittest.mock import MagicMock, patch

import pandas as pd

from pipelines.daily_market_data.analyst_recommendations_data import (
    AnalystRecommendationsData,
    AnalystUpgradesDowngradesData,
)


def test_analyst_recommendations_run_pulls_recommendations():
    pipeline = AnalystRecommendationsData(tickers=["AAPL"])
    recommendations = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=recommendations)

    out = pipeline.run()

    assert len(out) == 1
    assert (
        pipeline._pull_with_missing_ticker_retries.call_args.kwargs["client_method"]
        == "get_recommendations"
    )


def test_analyst_upgrades_run_pulls_upgrades_downgrades():
    pipeline = AnalystUpgradesDowngradesData(tickers=["AAPL"])
    upgrades = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "GRADEDATE": ["2024-01-01"]}
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=upgrades)

    out = pipeline.run()

    assert len(out) == 1
    assert (
        pipeline._pull_with_missing_ticker_retries.call_args.kwargs["client_method"]
        == "get_upgrades_downgrades"
    )


def test_analyst_recommendations_write_to_azure_writes_single_table():
    pipeline = AnalystRecommendationsData(tickers=["AAPL"])
    recommendations = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=recommendations)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    pipeline.azure_data_source.get_engine.assert_called_once_with(
        configs_path="config/configs.json"
    )
    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "analyst_recommendations"
    assert set(kwargs["dtype_overrides"]) == {"DATE"}


def test_analyst_upgrades_write_to_azure_writes_single_table():
    pipeline = AnalystUpgradesDowngradesData(tickers=["AAPL"])
    upgrades = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "GRADEDATE": ["2024-01-01"]}
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=upgrades)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True)

    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "analyst_upgrades_downgrades"
    assert set(kwargs["dtype_overrides"]) == {"DATE", "GRADEDATE"}


def test_analyst_recommendations_skip_write_when_duplicates_minus_date():
    pipeline = AnalystRecommendationsData(tickers=["AAPL"])
    recommendations = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "BUY": [10]}
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=recommendations)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(
        return_value=pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2024-01-01"], "BUY": [10]}
        )
    )
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True)

    pipeline.azure_data_source.write_sql_table.assert_not_called()


def test_analyst_recommendations_logs_warning_for_missing_tickers():
    pipeline = AnalystRecommendationsData(tickers=["AAPL", "MSFT"])
    recommendations = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=recommendations)

    with patch(
        "pipelines.daily_market_data.analyst_recommendations_data.log"
    ) as mock_log:
        _ = pipeline.run()

    warning_calls = [
        call for call in mock_log.call_args_list if call.kwargs.get("type") == "warning"
    ]
    assert (
        warning_calls
    ), "Expected a warning log for missing analyst recommendation tickers."
    assert "MSFT" in warning_calls[0].args[0]
