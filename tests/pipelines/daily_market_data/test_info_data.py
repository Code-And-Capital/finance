from unittest.mock import MagicMock, patch

import pandas as pd

from pipelines.daily_market_data.info_data import InfoData


def _build_batch_client(info_df: pd.DataFrame, officers_df: pd.DataFrame):
    client = MagicMock()
    client.get_company_info.return_value = info_df
    client.get_officer_info.return_value = officers_df
    return client


def test_run_batches_and_aggregates_info_and_officers():
    info = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "DATE": ["2024-01-02", "2024-01-02"],
            "MOSTRECENTQUARTER": ["2024-03-31", "2024-03-31"],
            "GOVERNANCEEPOCHDATE": ["2024-01-01", "2024-01-01"],
            "MISCLISTFIELD": [[1, 2, 3], "ok"],
        }
    )
    officers = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "DATE": ["2024-01-02", "2024-01-02"],
            "NAME": ["Tim Cook", "Satya Nadella"],
        }
    )
    pipeline = InfoData(tickers=["AAPL", "MSFT"])
    pipeline._create_client_for_tickers = MagicMock(
        return_value=_build_batch_client(info, officers)
    )

    info_df, officers_df = pipeline.run()

    pipeline._create_client_for_tickers.assert_called_once_with(["AAPL", "MSFT"])
    assert set(info_df["TICKER"]) == {"AAPL", "MSFT"}
    assert set(officers_df["TICKER"]) == {"AAPL", "MSFT"}
    assert pd.api.types.is_datetime64_any_dtype(info_df["MOSTRECENTQUARTER"])
    assert pd.api.types.is_datetime64_any_dtype(info_df["GOVERNANCEEPOCHDATE"])
    assert pd.isna(info_df.loc[info_df["TICKER"] == "AAPL", "MISCLISTFIELD"]).iloc[0]


def test_retry_creates_new_clients_for_remaining_batches():
    first_info = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})
    first_off = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "NAME": ["Tim Cook"]}
    )
    second_info = pd.DataFrame({"TICKER": ["MSFT"], "DATE": ["2024-01-02"]})
    second_off = pd.DataFrame(
        {"TICKER": ["MSFT"], "DATE": ["2024-01-02"], "NAME": ["Satya Nadella"]}
    )

    pipeline = InfoData(tickers=["AAPL", "MSFT"])
    pipeline._create_client_for_tickers = MagicMock(
        side_effect=[
            _build_batch_client(first_info, first_off),
            _build_batch_client(second_info, second_off),
        ]
    )

    with patch("pipelines.daily_market_data.info_data.time.sleep") as mock_sleep:
        info_df, officers_df = pipeline.run()

    assert pipeline._create_client_for_tickers.call_count == 2
    assert pipeline._create_client_for_tickers.call_args_list[0].args[0] == [
        "AAPL",
        "MSFT",
    ]
    assert pipeline._create_client_for_tickers.call_args_list[1].args[0] == ["MSFT"]
    mock_sleep.assert_called_once_with(120)
    assert set(info_df["TICKER"]) == {"AAPL", "MSFT"}
    assert set(officers_df["TICKER"]) == {"AAPL", "MSFT"}


def test_pull_officers_reuses_latest_results():
    info = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})
    officers = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "NAME": ["Tim Cook"]}
    )
    pipeline = InfoData(tickers=["AAPL"])
    pipeline._create_client_for_tickers = MagicMock(
        return_value=_build_batch_client(info, officers)
    )

    _ = pipeline.pull_info()
    out = pipeline.pull_officers()

    assert isinstance(out, pd.DataFrame)
    assert pipeline._create_client_for_tickers.call_count == 1


def test_logs_warning_after_max_resets():
    pipeline = InfoData(tickers=["AAPL", "MSFT"])
    pipeline.max_yfinance_resets = 1
    empty_client = _build_batch_client(pd.DataFrame(), pd.DataFrame())
    pipeline._create_client_for_tickers = MagicMock(return_value=empty_client)

    with patch("pipelines.daily_market_data.info_data.time.sleep") as mock_sleep, patch(
        "pipelines.daily_market_data.info_data.log"
    ) as mock_log:
        out = pipeline.pull_info()

    assert out.empty
    assert pipeline._create_client_for_tickers.call_count == 2
    assert mock_sleep.call_count == 1
    assert any(
        "Reached max yfinance resets" in call.args[0]
        for call in mock_log.call_args_list
    )


def test_run_default_does_not_write_to_azure():
    info = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})
    officers = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "NAME": ["Tim Cook"]}
    )
    pipeline = InfoData(tickers=["AAPL"])
    pipeline._create_client_for_tickers = MagicMock(
        return_value=_build_batch_client(info, officers)
    )
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run()

    pipeline.azure_data_source.write_sql_table.assert_not_called()


def test_run_write_to_azure_writes_company_info_and_officers():
    info = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})
    officers = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "NAME": ["Tim Cook"]}
    )
    pipeline = InfoData(tickers=["AAPL"])
    pipeline._create_client_for_tickers = MagicMock(
        return_value=_build_batch_client(info, officers)
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(
        side_effect=[
            pd.DataFrame(columns=["TICKER", "FIGI", "DATE", "NAME"]),
            pd.DataFrame(columns=["ADDRESS1", "CITY", "COUNTRY", "LAT", "LON"]),
        ]
    )
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    info_df, officers_df = pipeline.run(
        write_to_azure=True,
        configs_path="config/configs.json",
        ticker_to_figi={"AAPL": "FIGI_AAPL"},
    )

    pipeline.azure_data_source.get_engine.assert_called_once_with(
        configs_path="config/configs.json"
    )
    assert pipeline.azure_data_source.write_sql_table.call_count == 2
    first = pipeline.azure_data_source.write_sql_table.call_args_list[0].kwargs
    second = pipeline.azure_data_source.write_sql_table.call_args_list[1].kwargs
    assert first["table_name"] == "company_info"
    assert second["table_name"] == "officers"
    assert first["overwrite"] is False
    assert second["overwrite"] is False
    assert first["df"].equals(info_df)
    assert second["df"].equals(officers_df)


def test_run_write_to_azure_geocodes_and_writes_missing_addresses():
    info = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "ADDRESS1": ["1 Apple Park Way"],
            "CITY": ["Cupertino"],
            "COUNTRY": ["United States"],
        }
    )
    officers = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "NAME": ["Tim Cook"]}
    )
    pipeline = InfoData(tickers=["AAPL"])
    pipeline._create_client_for_tickers = MagicMock(
        return_value=_build_batch_client(info, officers)
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(
        side_effect=[
            pd.DataFrame(columns=["TICKER", "FIGI", "DATE", "NAME"]),
            pd.DataFrame(columns=["ADDRESS1", "CITY", "COUNTRY", "LAT", "LON"]),
        ]
    )
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    geocoded = pd.DataFrame(
        {
            "ADDRESS1": ["1 Apple Park Way"],
            "CITY": ["Cupertino"],
            "COUNTRY": ["United States"],
            "LAT": [37.3349],
            "LON": [-122.009],
        }
    )
    with patch(
        "pipelines.daily_market_data.info_data.geo.geocode_dataframe",
        return_value=geocoded,
    ):
        pipeline.run(write_to_azure=True, ticker_to_figi={"AAPL": "FIGI_AAPL"})

    assert pipeline.azure_data_source.write_sql_table.call_count == 3
    third = pipeline.azure_data_source.write_sql_table.call_args_list[2].kwargs
    assert third["table_name"] == "address"
    assert third["df"]["ADDRESS1"].iloc[0] == "1 Apple Park Way"


def test_run_write_to_azure_skips_geocode_when_no_missing_address():
    info = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "ADDRESS1": ["1 Apple Park Way"],
            "CITY": ["Cupertino"],
            "COUNTRY": ["United States"],
        }
    )
    officers = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "NAME": ["Tim Cook"]}
    )
    pipeline = InfoData(tickers=["AAPL"])
    pipeline._create_client_for_tickers = MagicMock(
        return_value=_build_batch_client(info, officers)
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(
        side_effect=[
            pd.DataFrame(columns=["TICKER", "FIGI", "DATE", "NAME"]),
            pd.DataFrame(
                {
                    "ADDRESS1": ["1 Apple Park Way"],
                    "CITY": ["Cupertino"],
                    "COUNTRY": ["United States"],
                    "LAT": [37.3349],
                    "LON": [-122.009],
                }
            ),
        ]
    )
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    with patch(
        "pipelines.daily_market_data.info_data.geo.geocode_dataframe"
    ) as mock_geocode:
        pipeline.run(write_to_azure=True, ticker_to_figi={"AAPL": "FIGI_AAPL"})

    mock_geocode.assert_not_called()
    assert pipeline.azure_data_source.write_sql_table.call_count == 2


def test_run_write_to_azure_skips_officer_write_when_rows_already_exist_minus_date():
    info = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})
    officers = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "NAME": ["Tim Cook"],
            "TITLE": ["CEO"],
        }
    )
    existing_officers = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "FIGI": ["FIGI_AAPL"],
            "DATE": ["2024-01-01"],
            "NAME": ["Tim Cook"],
            "TITLE": ["CEO"],
        }
    )
    pipeline = InfoData(tickers=["AAPL"])
    pipeline._create_client_for_tickers = MagicMock(
        return_value=_build_batch_client(info, officers)
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(
        side_effect=[
            existing_officers,
            pd.DataFrame(columns=["ADDRESS1", "CITY", "COUNTRY", "LAT", "LON"]),
        ]
    )
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True, ticker_to_figi={"AAPL": "FIGI_AAPL"})

    assert pipeline.azure_data_source.write_sql_table.call_count == 1
    only_call = pipeline.azure_data_source.write_sql_table.call_args_list[0].kwargs
    assert only_call["table_name"] == "company_info"
