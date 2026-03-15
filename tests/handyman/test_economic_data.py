import pandas as pd
import pytest

from handyman.economic_data import get_economic_data, get_economic_metadata


def test_get_economic_metadata_reads_fred_series_and_converts_dates(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_read_table_by_filters(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "TICKER": ["FEDFUNDS"],
                "TITLE": ["Federal Funds Effective Rate"],
                "OBSERVATION_START": ["1954-07-01"],
            }
        )

    monkeypatch.setattr(
        "handyman.economic_data.read_table_by_filters",
        fake_read_table_by_filters,
    )

    out = get_economic_metadata(
        tickers=["FEDFUNDS"],
        start_date="1954-01-01",
        end_date="1954-12-31",
    )

    assert captured["table_name"] == "fred_series"
    assert captured["tickers"] == ["FEDFUNDS"]
    assert captured["date_column"] == "OBSERVATION_START"
    assert captured["start_date"] == "1954-01-01"
    assert captured["end_date"] == "1954-12-31"
    assert pd.api.types.is_datetime64_any_dtype(out["OBSERVATION_START"])


def test_get_economic_data_reads_fred_table_and_converts_dates(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_read_table_by_filters(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "TICKER": ["FEDFUNDS", "FEDFUNDS"],
                "OBSERVATION_DATE": ["2025-03-01", "2025-03-01"],
                "REALTIME_START": ["2025-04-10", "2026-02-13"],
                "REALTIME_END": ["2026-02-12", "9999-12-31"],
                "VALUE": [4.33, 4.25],
            }
        )

    monkeypatch.setattr(
        "handyman.economic_data.read_table_by_filters",
        fake_read_table_by_filters,
    )

    out = get_economic_data(
        tickers=["FEDFUNDS"],
        start_date="2025-01-01",
        end_date="2025-12-31",
    )

    assert captured["table_name"] == "fred_economic_data"
    assert captured["tickers"] == ["FEDFUNDS"]
    assert captured["date_column"] == "OBSERVATION_DATE"
    assert captured["start_date"] == "2025-01-01"
    assert captured["end_date"] == "2025-12-31"
    assert pd.api.types.is_datetime64_any_dtype(out["OBSERVATION_DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["REALTIME_START"])
    assert pd.api.types.is_datetime64_any_dtype(out["REALTIME_END"])


def test_get_economic_data_missing_observation_date_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.economic_data.read_table_by_filters",
        lambda **_: pd.DataFrame({"TICKER": ["FEDFUNDS"], "VALUE": [4.33]}),
    )

    with pytest.raises(ValueError, match="Expected column 'OBSERVATION_DATE'"):
        get_economic_data(tickers=["FEDFUNDS"])
