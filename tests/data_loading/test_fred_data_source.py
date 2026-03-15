from unittest.mock import MagicMock

import pandas as pd
import pytest

from connectors.fred_data_source import FredDataClient


class DummyResponse:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self.payload


@pytest.fixture
def dummy_session() -> MagicMock:
    return MagicMock()


@pytest.fixture
def fred_config_path(tmp_path) -> str:
    config_path = tmp_path / "configs.json"
    config_path.write_text(
        '{"fred": {"api_key": "from-config"}}',
        encoding="utf-8",
    )
    return str(config_path)


def build_client(
    dummy_session: MagicMock,
    fred_config_path: str,
    **kwargs,
) -> FredDataClient:
    return FredDataClient(
        ["fedfunds", "UNRATE"],
        session=dummy_session,
        configs_path=fred_config_path,
        **kwargs,
    )


def test_init_normalizes_codes_and_sets_config(dummy_session, fred_config_path):
    client = FredDataClient(
        ["fedfunds", " FEDFUNDS ", "unrate"],
        session=dummy_session,
        configs_path=fred_config_path,
        max_workers=2,
        retries=4,
        timeout=15,
    )

    assert client.series_codes == ["FEDFUNDS", "UNRATE"]
    assert client.max_workers == 2
    assert client.retries == 4
    assert client.timeout == 15


def test_init_loads_api_key_from_configs_file(dummy_session, fred_config_path):
    client = FredDataClient(
        ["FEDFUNDS"],
        session=dummy_session,
        configs_path=fred_config_path,
    )

    assert client.api_key == "from-config"


@pytest.mark.parametrize(
    "series_codes,max_workers,retries,timeout,expected",
    [
        ("FEDFUNDS", 4, 3, 30, "series_codes must be a sequence"),
        ([], 4, 3, 30, "series_codes cannot be empty"),
        (["FEDFUNDS"], 0, 3, 30, "max_workers must be >= 1"),
        (["FEDFUNDS"], 4, 0, 30, "retries must be >= 1"),
        (["FEDFUNDS"], 4, 3, 0, "timeout must be >= 1"),
    ],
)
def test_init_validation_errors(
    dummy_session,
    fred_config_path,
    series_codes,
    max_workers,
    retries,
    timeout,
    expected,
):
    with pytest.raises(ValueError, match=expected):
        FredDataClient(
            series_codes,
            session=dummy_session,
            configs_path=fred_config_path,
            max_workers=max_workers,
            retries=retries,
            timeout=timeout,
        )


def test_get_series_metadata_normalizes_response(dummy_session, fred_config_path):
    dummy_session.get.return_value = DummyResponse(
        {
            "seriess": [
                {
                    "id": "FEDFUNDS",
                    "title": "Federal Funds Effective Rate",
                    "frequency": "Monthly",
                    "units": "Percent",
                    "seasonal_adjustment": "Not Seasonally Adjusted",
                    "observation_start": "1954-07-01",
                    "observation_end": "2024-01-01",
                    "realtime_start": "2024-02-01",
                    "realtime_end": "2024-02-01",
                    "last_updated": "2024-02-05 07:01:01-06",
                }
            ]
        }
    )
    client = build_client(dummy_session, fred_config_path)

    out = client.get_series_metadata("FEDFUNDS")

    assert out.loc[0, "TICKER"] == "FEDFUNDS"
    assert out.loc[0, "TITLE"] == "Federal Funds Effective Rate"
    assert str(out.loc[0, "OBSERVATION_START"]) == "1954-07-01"
    assert "LAST_METADATA_REFRESH_AT" not in out.columns
    assert "REALTIME_START" not in out.columns
    assert "REALTIME_END" not in out.columns
    assert "LAST_UPDATED" not in out.columns


def test_get_series_observations_normalizes_missing_values_and_dates(
    dummy_session,
    fred_config_path,
):
    dummy_session.get.return_value = DummyResponse(
        {
            "count": 2,
            "offset": 0,
            "limit": 100000,
            "observations": [
                {
                    "realtime_start": "2024-01-01",
                    "realtime_end": "2024-01-31",
                    "date": "2023-12-01",
                    "value": "5.33",
                },
                {
                    "realtime_start": "2024-02-01",
                    "realtime_end": "9999-12-31",
                    "date": "2023-12-01",
                    "value": ".",
                },
            ],
        }
    )
    client = build_client(dummy_session, fred_config_path)

    out = client.get_series_vintage_observations("FEDFUNDS")

    assert list(out["TICKER"].unique()) == ["FEDFUNDS"]
    assert len(out) == 1
    assert str(out.loc[0, "OBSERVATION_DATE"]) == "2023-12-01"
    assert out.loc[0, "VALUE"] == 5.33
    assert "IS_MISSING_VALUE" not in out.columns
    assert "VALUE_RAW" not in out.columns


def test_get_series_vintages_paginates(dummy_session, fred_config_path):
    dummy_session.get.side_effect = [
        DummyResponse(
            {
                "count": 3,
                "offset": 0,
                "limit": 2,
                "vintage_dates": ["2024-01-01", "2024-02-01"],
            }
        ),
        DummyResponse(
            {
                "count": 3,
                "offset": 2,
                "limit": 2,
                "vintage_dates": ["2024-03-01"],
            }
        ),
    ]
    client = build_client(dummy_session, fred_config_path)

    out = client.get_series_vintages("FEDFUNDS")

    assert len(out) == 3
    assert [str(value) for value in out["VINTAGE_DATE"]] == [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
    ]


def test_get_many_series_uses_per_series_options(
    dummy_session,
    fred_config_path,
    monkeypatch,
):
    client = build_client(dummy_session, fred_config_path)
    captured: list[tuple[str, dict]] = []

    def fake_get_series_vintage_observations(series_code: str, **kwargs):
        captured.append((series_code, kwargs))
        return pd.DataFrame(
            {
                "TICKER": [series_code],
                "OBSERVATION_DATE": [pd.Timestamp("2024-01-01").date()],
                "REALTIME_START": [pd.Timestamp("2024-02-01").date()],
                "REALTIME_END": [pd.Timestamp("9999-12-31").date()],
                "VALUE": [1.0],
            }
        )

    monkeypatch.setattr(
        client,
        "get_series_vintage_observations",
        fake_get_series_vintage_observations,
    )

    out = client.get_many_series(
        ["FEDFUNDS", "UNRATE"],
        series_options={"UNRATE": {"units": "pc1"}},
        observation_start="2023-01-01",
    )

    assert len(out) == 2
    assert captured[0][1]["observation_start"] == "2023-01-01"
    assert captured[1][1]["units"] == "pc1"


def test_request_json_retries_after_rate_limit_with_one_minute_sleep(
    dummy_session,
    fred_config_path,
    monkeypatch,
):
    dummy_session.get.side_effect = [
        DummyResponse({}, status_code=429),
        DummyResponse({"seriess": [{"id": "FEDFUNDS", "title": "Fed Funds"}]}),
    ]
    client = build_client(dummy_session, fred_config_path, retries=2)
    captured_sleep: list[float] = []
    captured_logs: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        "connectors.fred_data_source.time.sleep",
        lambda seconds: captured_sleep.append(seconds),
    )
    monkeypatch.setattr(
        "connectors.fred_data_source.log",
        lambda message, type=None: captured_logs.append((message, type)),
    )

    out = client.get_series_metadata("FEDFUNDS")

    assert out.loc[0, "TICKER"] == "FEDFUNDS"
    assert captured_sleep == [60.0]
    assert dummy_session.get.call_count == 2
    assert any("retrying after 60 seconds" in message for message, _ in captured_logs)
    assert any(level == "warning" for _, level in captured_logs)
