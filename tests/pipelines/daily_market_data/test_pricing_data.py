from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipelines.daily_market_data.pricing_data import PricingData


def test_run_requires_ticker_to_figi_when_start_mapping_enabled():
    pipeline = PricingData(tickers=["AAPL"])

    with pytest.raises(
        ValueError,
        match="ticker_to_figi must be provided when use_start_date_mapping=True",
    ):
        pipeline.run(use_start_date_mapping=True)


def test_run_default_without_start_mapping_returns_date_only():
    dummy_client = MagicMock()
    dummy_client.get_prices.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-02 10:00:00", "2024-01-03 10:00:00"],
            "TICKER": ["AAPL", "AAPL"],
            "ADJ_CLOSE": [150.0, 151.0],
        }
    )
    pipeline = PricingData(tickers=["AAPL"], client=dummy_client)

    result = pipeline.run(ticker_to_figi={"AAPL": "FIGI_AAPL"})

    assert isinstance(result, pd.DataFrame)
    assert "DATE" in result.columns
    assert str(result["DATE"].iloc[0]) == "2024-01-02"
    assert result["FIGI"].iloc[0] == "FIGI_AAPL"


def test_build_max_date_query_uses_figi_filter():
    pipeline = PricingData(tickers=["AAPL"])

    query = pipeline._build_max_date_query(figi_values=["FIGI_AAPL", "FIGI_MSFT"])

    assert "FIGI" in query
    assert "FIGI_AAPL" in query
    assert "FIGI_MSFT" in query


def test_build_start_date_mapping_maps_figi_dates_back_to_tickers():
    pipeline = PricingData(tickers=["AAPL", "MSFT"])
    client = MagicMock()
    client.tickers = ["AAPL", "MSFT"]
    max_dates = pd.DataFrame(
        {
            "FIGI": ["FIGI_AAPL"],
            "START_DATE": [pd.Timestamp("2024-01-02")],
        }
    )

    mapping = pipeline._build_start_date_mapping(
        max_dates=max_dates,
        yahoo_client=client,
        ticker_to_figi={"AAPL": "FIGI_AAPL", "MSFT": "FIGI_MSFT"},
    )

    assert pd.Timestamp(mapping["AAPL"]).date() == pd.Timestamp("2024-01-03").date()
    assert pd.Timestamp(mapping["MSFT"]).date() == pd.Timestamp("2000-01-01").date()


def test_run_skips_future_start_dates_and_pulls_subset():
    pipeline = PricingData(tickers=["AAPL", "MSFT"])
    today = pd.Timestamp.today().normalize().date()
    future_day = pd.Timestamp(today + pd.Timedelta(days=1))

    root_client = MagicMock()
    root_client.tickers = ["AAPL", "MSFT"]
    subset_client = MagicMock()
    pipeline._resolve_client = MagicMock(return_value=root_client)
    pipeline._create_client_for_tickers = MagicMock(return_value=subset_client)
    pipeline._fetch_max_dates = MagicMock(
        return_value=pd.DataFrame({"FIGI": [], "START_DATE": []})
    )
    pipeline._build_start_date_mapping = MagicMock(
        return_value={"AAPL": pd.Timestamp(today), "MSFT": future_day}
    )
    pipeline._pull_generic = MagicMock(return_value=pd.DataFrame())
    pipeline._today = MagicMock(return_value=today)

    pipeline.run(
        use_start_date_mapping=True,
        ticker_to_figi={"AAPL": "FIGI_AAPL", "MSFT": "FIGI_MSFT"},
    )

    pipeline._create_client_for_tickers.assert_called_once_with(["AAPL"])
    method_kwargs = pipeline._pull_generic.call_args.kwargs["method_kwargs"]
    assert set(method_kwargs["start_date"]) == {"AAPL"}


@patch("pipelines.daily_market_data.pricing_data.log")
def test_run_overwrites_adjusted_figis(mock_log):
    dummy_client = MagicMock()
    dummy_client.get_prices.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-02", "2024-01-03"],
            "TICKER": ["AAPL", "MSFT"],
            "FIGI": ["FIGI_AAPL", "FIGI_MSFT"],
            "DIVIDENDS": [0.1, 0.0],
            "STOCK_SPLITS": [0.0, 0.0],
            "ADJ_CLOSE": [150.0, 300.0],
        }
    )
    pipeline = PricingData(tickers=["AAPL", "MSFT"], client=dummy_client)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)
    pipeline._overwrite_adjusted_figis = MagicMock(return_value=None)

    pipeline.run(
        write_to_azure=True,
        adjust_for_corporate_actions=True,
        ticker_to_figi={"AAPL": "FIGI_AAPL", "MSFT": "FIGI_MSFT"},
    )

    pipeline._overwrite_adjusted_figis.assert_called_once()
    called_adjusted = pipeline._overwrite_adjusted_figis.call_args.kwargs[
        "adjusted_figis"
    ]
    assert called_adjusted == ["FIGI_AAPL"]
    assert any(
        "Adjusted FIGIs identified for full overwrite" in call.args[0]
        for call in mock_log.call_args_list
    )


def test_run_write_to_azure_writes_prices():
    dummy_client = MagicMock()
    dummy_client.get_prices.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-02", "2024-01-03"],
            "TICKER": ["AAPL", "AAPL"],
            "ADJ_CLOSE": [150.0, 151.0],
        }
    )
    pipeline = PricingData(tickers=["AAPL"], client=dummy_client)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    result = pipeline.run(
        write_to_azure=True,
        configs_path="config/configs.json",
        ticker_to_figi={"AAPL": "FIGI_AAPL"},
    )

    pipeline.azure_data_source.get_engine.assert_called_once_with(
        configs_path="config/configs.json"
    )
    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "prices"
    assert kwargs["overwrite"] is False
    assert kwargs["df"].equals(result)


@patch(
    "pipelines.daily_market_data.pricing_data.dataframe_utils.ensure_datetime_column"
)
def test_run_empty_dataframe_skips_type_coercion(mock_ensure_datetime):
    dummy_client = MagicMock()
    dummy_client.get_prices.return_value = pd.DataFrame()
    pipeline = PricingData(tickers=["AAPL"], client=dummy_client)

    result = pipeline.run(ticker_to_figi={"AAPL": "FIGI_AAPL"})

    assert result.empty
    mock_ensure_datetime.assert_not_called()
