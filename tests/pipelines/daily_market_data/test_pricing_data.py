from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from pipelines.daily_market_data.pricing_data import PricingData


@patch("pipelines.daily_market_data.pricing_data.dataframe_utils.df_to_dict")
def test_pricing_data_run_happy_path(mock_df_to_dict):
    pipeline = PricingData(tickers=["AAPL", "MSFT"])

    mock_client = MagicMock()
    mock_client.tickers = ["AAPL", "MSFT"]
    mock_client.get_prices.return_value = pd.DataFrame(
        {"DATE": ["2024-01-02", "2024-01-03"], "AAPL": [100, 101], "MSFT": [200, 201]}
    )
    pipeline._create_client = MagicMock(return_value=mock_client)

    mock_read_sql = MagicMock(
        return_value=pd.DataFrame(
            {"TICKER": ["AAPL", "MSFT"], "START_DATE": ["2024-01-01", "2024-01-02"]}
        )
    )
    pipeline.azure_data_source.read_sql_table = mock_read_sql
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())

    mock_df_to_dict.return_value = {
        "AAPL": pd.Timestamp("2024-01-02"),
        "MSFT": pd.Timestamp("2024-01-03"),
    }

    result = pipeline.run(use_start_date_mapping=True)

    mock_read_sql.assert_called_once()
    mock_df_to_dict.assert_called_once()
    mock_client.get_prices.assert_called_once()
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) >= {"DATE", "AAPL", "MSFT"}
    assert result.loc[0, "AAPL"] == 100


def test_pricing_data_run_default_without_start_date_mapping():
    dummy_client = MagicMock()
    dummy_client.get_prices.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-02", "2024-01-03"],
            "AAPL": [150.0, 151.0],
        }
    )

    pipeline = PricingData(tickers=["AAPL"], client=dummy_client)

    result = pipeline.run()

    dummy_client.get_prices.assert_called_once_with()
    assert isinstance(result, pd.DataFrame)
    assert "DATE" in result.columns


def test_pricing_data_run_default_does_not_write_to_azure():
    dummy_client = MagicMock()
    dummy_client.get_prices.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-02", "2024-01-03"],
            "AAPL": [150.0, 151.0],
        }
    )

    pipeline = PricingData(tickers=["AAPL"], client=dummy_client)
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run()
    pipeline.azure_data_source.write_sql_table.assert_not_called()


def test_pricing_data_run_write_to_azure_writes_prices():
    dummy_client = MagicMock()
    dummy_client.get_prices.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-02", "2024-01-03"],
            "AAPL": [150.0, 151.0],
        }
    )

    pipeline = PricingData(tickers=["AAPL"], client=dummy_client)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    result = pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    pipeline.azure_data_source.get_engine.assert_called_once_with(
        configs_path="config/configs.json"
    )
    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "prices"
    assert kwargs["overwrite"] is False
    assert kwargs["df"].equals(result)
    assert str(kwargs["df"]["DATE"].iloc[0]) == "2024-01-02"


@patch("pipelines.daily_market_data.pricing_data.log")
def test_pricing_data_run_overwrites_adjusted_tickers(mock_log):
    dummy_client = MagicMock()
    dummy_client.get_prices.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-02", "2024-01-03"],
            "TICKER": ["AAPL", "MSFT"],
            "DIVIDENDS": [0.1, 0.0],
            "STOCK_SPLITS": [0.0, 0.0],
            "ADJ_CLOSE": [150.0, 300.0],
        }
    )

    pipeline = PricingData(tickers=["AAPL", "MSFT"], client=dummy_client)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)
    pipeline.azure_data_source.delete_sql_rows = MagicMock(return_value=None)
    pipeline._pull_adjusted_prices = MagicMock(
        return_value=pd.DataFrame(
            {
                "DATE": [pd.to_datetime("2024-01-02").date()],
                "TICKER": ["AAPL"],
                "DIVIDENDS": [0.0],
                "STOCK_SPLITS": [0.0],
                "ADJ_CLOSE": [149.0],
            }
        )
    )

    pipeline.run(write_to_azure=True, adjust_for_corporate_actions=True)

    pipeline.azure_data_source.delete_sql_rows.assert_called_once()
    pipeline._pull_adjusted_prices.assert_called_once()
    assert pipeline.azure_data_source.write_sql_table.call_count == 2
    assert any(
        "Adjusted tickers identified for full overwrite" in call.args[0]
        for call in mock_log.call_args_list
    )


@patch("pipelines.daily_market_data.pricing_data.dataframe_utils.df_to_dict")
def test_pricing_data_run_with_array_like_input(mock_df_to_dict):
    tickers = np.array(["AAPL", "MSFT"])
    dummy_client = MagicMock()
    dummy_client.tickers = ["AAPL", "MSFT"]
    dummy_client.get_prices.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "DATE": ["2024-01-02", "2024-01-03"],
            "ADJ_CLOSE": [150.0, 300.0],
        }
    )

    pipeline = PricingData(tickers=tickers, client=dummy_client)
    pipeline.azure_data_source.read_sql_table = MagicMock(
        return_value=pd.DataFrame(
            {"TICKER": ["AAPL", "MSFT"], "START_DATE": ["2024-01-01", "2024-01-02"]}
        )
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())

    mock_df_to_dict.return_value = {
        "AAPL": pd.Timestamp("2024-01-02"),
        "MSFT": pd.Timestamp("2024-01-03"),
    }

    result = pipeline.run(use_start_date_mapping=True)
    assert isinstance(result, pd.DataFrame)


@patch("pipelines.daily_market_data.pricing_data.dataframe_utils.df_to_dict")
def test_pricing_data_run_missing_date_column_raises(mock_df_to_dict):
    dummy_client = MagicMock()
    dummy_client.tickers = ["AAPL"]
    dummy_client.get_prices.return_value = pd.DataFrame({"AAPL": [100.0]})

    pipeline = PricingData(tickers=["AAPL"], client=dummy_client)
    pipeline.azure_data_source.read_sql_table = MagicMock(
        return_value=pd.DataFrame({"TICKER": ["AAPL"], "START_DATE": ["2024-01-01"]})
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())

    mock_df_to_dict.return_value = {"AAPL": pd.Timestamp("2024-01-02")}

    import pytest

    with pytest.raises(ValueError, match="Expected column 'DATE'"):
        pipeline.run(use_start_date_mapping=True)


@patch(
    "pipelines.daily_market_data.pricing_data.dataframe_utils.ensure_datetime_column"
)
def test_pricing_data_run_empty_dataframe_skips_type_coercion(mock_ensure_datetime):
    dummy_client = MagicMock()
    dummy_client.get_prices.return_value = pd.DataFrame()

    pipeline = PricingData(tickers=["AAPL"], client=dummy_client)

    result = pipeline.run()

    assert result.empty
    mock_ensure_datetime.assert_not_called()


def test_pricing_data_skips_future_start_dates_and_pulls_subset():
    pipeline = PricingData(tickers=["AAPL", "MSFT"])
    today = pd.Timestamp.today().normalize().date()
    future_day = pd.Timestamp(today + pd.Timedelta(days=1))

    root_client = MagicMock()
    subset_client = MagicMock()
    pipeline._resolve_client = MagicMock(return_value=root_client)
    pipeline._create_client_for_tickers = MagicMock(return_value=subset_client)
    pipeline._fetch_max_dates = MagicMock(return_value=pd.DataFrame())
    pipeline._build_start_date_mapping = MagicMock(
        return_value={"AAPL": pd.Timestamp(today), "MSFT": future_day}
    )
    pipeline._pull_generic = MagicMock(return_value=pd.DataFrame())
    pipeline._today = MagicMock(return_value=today)

    pipeline.run(use_start_date_mapping=True)

    pipeline._create_client_for_tickers.assert_called_once_with(["AAPL"])
    pull_kwargs = pipeline._pull_generic.call_args.kwargs["method_kwargs"]
    assert set(pull_kwargs["start_date"].keys()) == {"AAPL"}


def test_add_missing_tickers_adds_missing_rows():
    pipeline = PricingData(tickers=["AAPL"])
    df = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "START_DATE": ["2010-01-01", "2011-01-01"],
        }
    )
    result = pipeline._add_missing_tickers(df, ["AAPL", "MSFT", "GOOG"])
    assert set(result["TICKER"]) == {"AAPL", "MSFT", "GOOG"}
    assert result.loc[result["TICKER"] == "GOOG", "START_DATE"].iloc[0] == "2000-01-01"


def test_add_missing_tickers_missing_ticker_column_raises():
    import pytest

    pipeline = PricingData(tickers=["AAPL"])
    df = pd.DataFrame({"START_DATE": ["2010-01-01"]})
    with pytest.raises(KeyError):
        pipeline._add_missing_tickers(df, ["AAPL"])
