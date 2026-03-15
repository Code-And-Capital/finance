from decimal import Decimal
from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from pipelines.daily_market_data.fred_data import FredData


def _metadata_df(
    *,
    tickers: list[str],
    frequencies: list[str],
    observation_starts: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TICKER": tickers,
            "TITLE": tickers,
            "FREQUENCY": frequencies,
            "OBSERVATION_START": [
                pd.Timestamp(value).date() for value in observation_starts
            ],
        }
    )


def test_get_series_config_dataframe_uses_defaults_and_filters_requested_codes():
    pipeline = FredData(series_codes=["FEDFUNDS", "UNRATE"])
    pipeline._load_fred_settings = MagicMock(return_value={"api_key": "fred-key"})

    out = pipeline._get_series_config_dataframe()

    assert set(out["TICKER"]) == {"FEDFUNDS", "UNRATE"}
    assert out["IS_ACTIVE"].all()


def test_filter_new_rows_skips_existing_duplicates():
    incoming = pd.DataFrame(
        {
            "TICKER": ["FEDFUNDS", "FEDFUNDS", "FEDFUNDS"],
            "OBSERVATION_DATE": [
                pd.Timestamp("2024-01-01").date(),
                pd.Timestamp("2024-02-01").date(),
                pd.Timestamp("2024-02-01").date(),
            ],
            "VALUE": [1.0, 2.0, 2.0],
        }
    )
    existing = pd.DataFrame(
        {
            "TICKER": ["FEDFUNDS"],
            "OBSERVATION_DATE": [pd.Timestamp("2024-01-01").date()],
            "VALUE": [1.0],
        }
    )

    filtered, skipped = FredData._filter_new_rows(
        incoming,
        existing,
        compare_columns=["TICKER", "OBSERVATION_DATE", "VALUE"],
    )

    assert len(filtered) == 1
    assert str(filtered.loc[0, "OBSERVATION_DATE"]) == "2024-02-01"
    assert skipped == 2


def test_run_uses_observation_start_for_revised_series_without_writes():
    pipeline = FredData(series_configs=[{"series_code": "FEDFUNDS"}])
    pipeline._load_fred_settings = MagicMock(
        return_value={"api_key": "fred-key", "full_history_batch_years": 100}
    )
    client = MagicMock()
    client.get_many_series_metadata.return_value = _metadata_df(
        tickers=["FEDFUNDS"],
        frequencies=["Monthly"],
        observation_starts=["1954-07-01"],
    )
    client.get_many_series.side_effect = [
        pd.DataFrame(
            {
                "TICKER": ["FEDFUNDS"],
                "OBSERVATION_DATE": ["2025-01-01"],
                "VALUE": [4.25],
                "REALTIME_START": ["2026-02-01"],
                "REALTIME_END": ["9999-12-31"],
            }
        ),
    ]
    pipeline.fred_client = client

    revised_data = pipeline.run(
        full_history=False,
        observation_start="2025-01-01",
        observation_end="2026-12-31",
        write_to_azure=False,
    )

    assert len(revised_data) == 1
    first_call = client.get_many_series.call_args_list[0].kwargs
    assert first_call["observation_start"] == "2025-01-01"
    assert first_call["observation_end"] == "2026-12-31"
    assert first_call["include_revisions"] is True
    assert set(revised_data["OBSERVATION_DATE"].astype(str)) == {"2025-01-01"}


def test_run_full_history_uses_metadata_start_for_revised_series():
    pipeline = FredData(series_configs=[{"series_code": "FEDFUNDS"}])
    pipeline._load_fred_settings = MagicMock(
        return_value={"api_key": "fred-key", "full_history_batch_years": 1}
    )
    pipeline._today = lambda: date(2025, 3, 14)
    client = MagicMock()
    client.get_many_series_metadata.return_value = _metadata_df(
        tickers=["FEDFUNDS"],
        frequencies=["Monthly"],
        observation_starts=["2024-01-01"],
    )
    client.get_many_series.side_effect = [
        pd.DataFrame(
            {
                "TICKER": ["FEDFUNDS"],
                "OBSERVATION_DATE": ["2024-01-01"],
                "VALUE": [5.0],
                "REALTIME_START": ["2024-02-01"],
                "REALTIME_END": ["9999-12-31"],
            }
        ),
        pd.DataFrame(
            {
                "TICKER": ["FEDFUNDS"],
                "OBSERVATION_DATE": ["2025-01-01"],
                "VALUE": [4.5],
                "REALTIME_START": ["2025-02-01"],
                "REALTIME_END": ["9999-12-31"],
            }
        ),
    ]
    pipeline.fred_client = client

    revised_data = pipeline.run(full_history=True, write_to_azure=False)

    assert len(revised_data) == 2
    first_call = client.get_many_series.call_args_list[0].kwargs
    second_call = client.get_many_series.call_args_list[1].kwargs
    assert first_call["observation_start"] == "2024-01-01"
    assert first_call["observation_end"] == "2024-12-31"
    assert second_call["observation_start"] == "2025-01-01"
    assert second_call["observation_end"] == "2025-03-14"


def test_refresh_metadata_then_run_reuses_cached_metadata():
    pipeline = FredData(series_configs=[{"series_code": "FEDFUNDS"}])
    pipeline._load_fred_settings = MagicMock(
        return_value={"api_key": "fred-key", "full_history_batch_years": 1}
    )
    pipeline._today = lambda: date(2025, 3, 14)
    client = MagicMock()
    client.get_many_series_metadata.return_value = _metadata_df(
        tickers=["FEDFUNDS"],
        frequencies=["Monthly"],
        observation_starts=["2024-01-01"],
    )
    client.get_many_series.return_value = pd.DataFrame(
        {
            "TICKER": ["FEDFUNDS"],
            "OBSERVATION_DATE": ["2024-01-01"],
            "VALUE": [5.0],
            "REALTIME_START": ["2024-02-01"],
            "REALTIME_END": ["9999-12-31"],
        }
    )
    pipeline.fred_client = client

    metadata = pipeline.refresh_metadata(write_to_azure=False)
    revised_data = pipeline.run(full_history=True, write_to_azure=False)

    assert not metadata.empty
    assert not revised_data.empty
    assert client.get_many_series_metadata.call_count == 1


def test_run_daily_series_uses_latest_only_full_history_and_stabilizes_realtime_dates():
    pipeline = FredData(series_configs=[{"series_code": "DFF"}])
    pipeline._load_fred_settings = MagicMock(
        return_value={"api_key": "fred-key", "full_history_batch_years": 1}
    )
    pipeline._today = lambda: date(2025, 3, 14)
    client = MagicMock()
    client.get_many_series_metadata.return_value = _metadata_df(
        tickers=["DFF"],
        frequencies=["Daily"],
        observation_starts=["2024-01-01"],
    )
    client.get_many_series.side_effect = [
        pd.DataFrame(
            {
                "TICKER": ["DFF"],
                "OBSERVATION_DATE": ["2024-01-01"],
                "VALUE": [5.1],
                "REALTIME_START": ["2025-03-14"],
                "REALTIME_END": ["2025-03-14"],
            }
        ),
        pd.DataFrame(
            {
                "TICKER": ["DFF"],
                "OBSERVATION_DATE": ["2025-01-02"],
                "VALUE": [4.9],
                "REALTIME_START": ["2025-03-14"],
                "REALTIME_END": ["2025-03-14"],
            }
        ),
    ]
    pipeline.fred_client = client

    out = pipeline.run(
        full_history=False,
        observation_start="2025-01-01",
        observation_end="2025-03-14",
        write_to_azure=False,
    )

    assert len(out) == 2
    first_call = client.get_many_series.call_args_list[0].kwargs
    assert first_call["include_revisions"] is False
    assert first_call["observation_start"] == "2024-01-01"
    assert str(out.loc[0, "REALTIME_START"]) == str(out.loc[0, "OBSERVATION_DATE"])
    assert str(out.loc[0, "REALTIME_END"]) == str(out.loc[0, "OBSERVATION_DATE"])


def test_run_splits_daily_and_revised_tickers_by_metadata_frequency():
    pipeline = FredData(
        series_configs=[{"series_code": "FEDFUNDS"}, {"series_code": "DFF"}]
    )
    pipeline._load_fred_settings = MagicMock(
        return_value={"api_key": "fred-key", "full_history_batch_years": 100}
    )
    client = MagicMock()
    client.get_many_series_metadata.return_value = _metadata_df(
        tickers=["FEDFUNDS", "DFF"],
        frequencies=["Monthly", "Daily"],
        observation_starts=["1954-07-01", "1954-07-01"],
    )
    client.get_many_series.side_effect = [
        pd.DataFrame(
            {
                "TICKER": ["FEDFUNDS"],
                "OBSERVATION_DATE": ["2025-01-01"],
                "VALUE": [4.25],
                "REALTIME_START": ["2026-02-01"],
                "REALTIME_END": ["9999-12-31"],
            }
        ),
        pd.DataFrame(
            {
                "TICKER": ["DFF"],
                "OBSERVATION_DATE": ["2025-01-02"],
                "VALUE": [4.9],
                "REALTIME_START": ["2026-03-14"],
                "REALTIME_END": ["2026-03-14"],
            }
        ),
    ]
    pipeline.fred_client = client

    out = pipeline.run(
        full_history=False,
        observation_start="2025-01-01",
        observation_end="2026-12-31",
        write_to_azure=False,
    )

    assert len(out) == 2
    revised_call = client.get_many_series.call_args_list[0].kwargs
    daily_call = client.get_many_series.call_args_list[1].kwargs
    assert revised_call["include_revisions"] is True
    assert daily_call["include_revisions"] is False


def test_run_write_to_azure_writes_only_revised_data_table():
    pipeline = FredData(series_configs=[{"series_code": "FEDFUNDS"}])
    pipeline._load_fred_settings = MagicMock(return_value={"api_key": "fred-key"})
    pipeline._table_exists = MagicMock(return_value=True)
    client = MagicMock()
    client.get_many_series_metadata.return_value = _metadata_df(
        tickers=["FEDFUNDS"],
        frequencies=["Monthly"],
        observation_starts=["1954-07-01"],
    )
    client.get_many_series.return_value = pd.DataFrame(
        {
            "TICKER": ["FEDFUNDS"],
            "OBSERVATION_DATE": ["2024-01-01"],
            "VALUE": [5.33],
            "REALTIME_START": ["2024-02-01"],
            "REALTIME_END": ["9999-12-31"],
        }
    )
    pipeline.fred_client = client
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(
        return_value=pd.DataFrame(
            {
                "TICKER": ["FEDFUNDS"],
                "OBSERVATION_DATE": ["2024-01-01"],
                "VALUE": [5.33],
                "REALTIME_START": ["2024-02-01"],
                "REALTIME_END": ["9999-12-31"],
            }
        )
    )
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)
    pipeline.azure_data_source.delete_sql_rows = MagicMock(return_value=None)

    revised_data = pipeline.run(
        write_to_azure=True,
        observation_start="2024-01-01",
        observation_end="2024-12-31",
    )

    assert revised_data.empty
    written_tables = [
        call.kwargs["table_name"]
        for call in pipeline.azure_data_source.write_sql_table.call_args_list
    ]
    assert written_tables == []


def test_run_write_to_azure_replaces_only_changed_observation_groups():
    pipeline = FredData(series_configs=[{"series_code": "CPIAUCSL"}])
    pipeline._load_fred_settings = MagicMock(return_value={"api_key": "fred-key"})
    pipeline._table_exists = MagicMock(return_value=True)
    client = MagicMock()
    client.get_many_series_metadata.return_value = _metadata_df(
        tickers=["CPIAUCSL"],
        frequencies=["Monthly"],
        observation_starts=["1947-01-01"],
    )
    client.get_many_series.return_value = pd.DataFrame(
        {
            "TICKER": ["CPIAUCSL", "CPIAUCSL", "CPIAUCSL"],
            "OBSERVATION_DATE": ["2025-03-01", "2025-03-01", "2025-04-01"],
            "VALUE": [319.615, 320.111, 321.0],
            "REALTIME_START": ["2025-04-10", "2026-02-13", "2025-05-13"],
            "REALTIME_END": ["2026-02-12", "9999-12-31", "9999-12-31"],
        }
    )
    pipeline.fred_client = client
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(
        return_value=pd.DataFrame(
            {
                "TICKER": ["CPIAUCSL", "CPIAUCSL", "CPIAUCSL"],
                "OBSERVATION_DATE": ["2025-03-01", "2025-03-01", "2025-04-01"],
                "VALUE": [319.615, 321.0, 321.0],
                "REALTIME_START": ["2025-04-10", "2025-05-13", "2025-05-13"],
                "REALTIME_END": ["9999-12-31", "9999-12-31", "9999-12-31"],
            }
        )
    )
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)
    pipeline.azure_data_source.delete_sql_rows = MagicMock(return_value=None)

    revised_data = pipeline.run(
        write_to_azure=True,
        observation_start="2026-01-01",
        observation_end="2026-12-31",
    )

    assert len(revised_data) == 2
    assert set(revised_data["OBSERVATION_DATE"].astype(str)) == {"2025-03-01"}
    delete_query = pipeline.azure_data_source.delete_sql_rows.call_args.kwargs["query"]
    assert "2025-03-01" in delete_query
    assert "2025-04-01" not in delete_query


def test_run_write_to_azure_skips_unchanged_groups_after_normalizing_existing_rows():
    pipeline = FredData(series_configs=[{"series_code": "FEDFUNDS"}])
    pipeline._load_fred_settings = MagicMock(return_value={"api_key": "fred-key"})
    pipeline._table_exists = MagicMock(return_value=True)
    client = MagicMock()
    client.get_many_series_metadata.return_value = _metadata_df(
        tickers=["FEDFUNDS"],
        frequencies=["Monthly"],
        observation_starts=["1954-07-01"],
    )
    client.get_many_series.return_value = pd.DataFrame(
        {
            "TICKER": ["FEDFUNDS"],
            "OBSERVATION_DATE": ["2025-03-01"],
            "VALUE": [4.33],
            "REALTIME_START": ["2025-04-10"],
            "REALTIME_END": ["9999-12-31"],
        }
    )
    pipeline.fred_client = client
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(
        return_value=pd.DataFrame(
            {
                "TICKER": ["fedfunds"],
                "OBSERVATION_DATE": [pd.Timestamp("2025-03-01")],
                "VALUE": [Decimal("4.33")],
                "REALTIME_START": [pd.Timestamp("2025-04-10")],
                "REALTIME_END": [pd.Timestamp("9999-12-31")],
            }
        )
    )
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)
    pipeline.azure_data_source.delete_sql_rows = MagicMock(return_value=None)

    revised_data = pipeline.run(
        write_to_azure=True,
        observation_start="2025-01-01",
        observation_end="2025-12-31",
    )

    assert revised_data.empty
    pipeline.azure_data_source.delete_sql_rows.assert_not_called()
    pipeline.azure_data_source.write_sql_table.assert_not_called()


def test_refresh_metadata_returns_metadata_without_observation_end():
    pipeline = FredData(series_configs=[{"series_code": "FEDFUNDS"}])
    pipeline._load_fred_settings = MagicMock(return_value={"api_key": "fred-key"})
    client = MagicMock()
    client.get_many_series_metadata.return_value = pd.DataFrame(
        {
            "TICKER": ["FEDFUNDS"],
            "TITLE": ["Federal Funds Effective Rate"],
            "FREQUENCY": ["Monthly"],
            "UNITS": ["Percent"],
            "SEASONAL_ADJUSTMENT": ["Not Seasonally Adjusted"],
            "OBSERVATION_START": [pd.Timestamp("1954-07-01").date()],
            "OBSERVATION_END": [pd.Timestamp("2024-12-01").date()],
        }
    )
    pipeline.fred_client = client

    metadata = pipeline.refresh_metadata(write_to_azure=False)

    assert "OBSERVATION_END" not in metadata.columns
    assert "REALTIME_START" not in metadata.columns
    assert "REALTIME_END" not in metadata.columns
    assert "LAST_UPDATED" not in metadata.columns
    assert metadata.loc[0, "TICKER"] == "FEDFUNDS"
