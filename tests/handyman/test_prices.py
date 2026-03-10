from __future__ import annotations

import pandas as pd
import pytest

from handyman.prices import get_analyst_price_targets, get_prices


@pytest.fixture
def sample_prices_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "TICKER": ["AAPL", "MSFT", "AAPL"],
            "ADJ_CLOSE": [100.0, 200.0, 101.0],
        }
    )


def test_get_prices_returns_long_dataframe(
    monkeypatch: pytest.MonkeyPatch, sample_prices_df: pd.DataFrame
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return sample_prices_df

    monkeypatch.setattr("handyman.prices.run_sql_template", fake_run_sql_template)

    result = get_prices(tickers=["AAPL", "MSFT"], start_date="2024-01-01")

    assert captured["sql_file"] == "prices.txt"
    filters = captured["filters"]
    assert "AAPL" in filters["security_filter"]
    assert "MSFT" in filters["security_filter"]
    assert filters["date_filter"] == "AND DATE >= '2024-01-01'"

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"DATE", "TICKER", "ADJ_CLOSE"}
    assert pd.api.types.is_datetime64_any_dtype(result["DATE"])
    assert len(result) == 3


def test_get_prices_supports_figi_filter(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return pd.DataFrame(
            {
                "DATE": ["2024-01-01"],
                "TICKER": ["AAPL"],
                "ADJ_CLOSE": [100.0],
            }
        )

    monkeypatch.setattr("handyman.prices.run_sql_template", fake_run_sql_template)

    _ = get_prices(figis=["BBG000B9XRY4"])
    assert "FIGI" in captured["filters"]["security_filter"]
    assert "BBG000B9XRY4" in captured["filters"]["security_filter"]


def test_get_prices_rejects_tickers_and_figis_together():
    with pytest.raises(ValueError, match="Provide only one of `tickers` or `figis`"):
        get_prices(tickers=["AAPL"], figis=["BBG000B9XRY4"])


def test_get_prices_includes_end_date_filter(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return pd.DataFrame(
            {
                "DATE": ["2024-01-01"],
                "TICKER": ["AAPL"],
                "ADJ_CLOSE": [100.0],
            }
        )

    monkeypatch.setattr("handyman.prices.run_sql_template", fake_run_sql_template)

    _ = get_prices(tickers=["AAPL"], start_date="2024-01-01", end_date="2024-01-31")

    assert "DATE >= '2024-01-01'" in captured["filters"]["date_filter"]
    assert "DATE <= '2024-01-31'" in captured["filters"]["date_filter"]


@pytest.mark.parametrize(
    "raw,error_pattern",
    [
        (
            pd.DataFrame({"TICKER": ["AAPL"], "ADJ_CLOSE": [1.0]}),
            "Expected column 'DATE'",
        ),
        (
            pd.DataFrame({"DATE": ["2024-01-01"], "ADJ_CLOSE": [1.0]}),
            "Expected columns",
        ),
        (
            pd.DataFrame({"DATE": ["2024-01-01"], "TICKER": ["AAPL"]}),
            "Expected columns",
        ),
    ],
)
def test_get_prices_missing_required_columns_raise(
    monkeypatch: pytest.MonkeyPatch,
    raw: pd.DataFrame,
    error_pattern: str,
):
    monkeypatch.setattr("handyman.prices.run_sql_template", lambda **_: raw)

    with pytest.raises(ValueError, match=error_pattern):
        get_prices(tickers=["AAPL"])


def test_get_prices_invalid_ticker_input_raises():
    with pytest.raises(TypeError):
        get_prices(tickers=["AAPL", 123])


def test_get_analyst_price_targets_builds_filters(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "TARGET_MEAN": [250.0]}
        )

    monkeypatch.setattr("handyman.prices.run_sql_template", fake_run_sql_template)

    out = get_analyst_price_targets(tickers=["AAPL"], start_date="2024-01-01")

    assert captured["sql_file"] == "base_with_filters.txt"
    assert "AAPL" in captured["filters"]["filters_sql"]
    assert "DATE >= '2024-01-01'" in captured["filters"]["filters_sql"]
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_analyst_price_targets_latest_uses_template_and_ignores_start_date(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "TARGET_MEAN": [250.0]}
        )

    monkeypatch.setattr("handyman.prices.run_sql_template", fake_run_sql_template)

    out = get_analyst_price_targets(
        tickers=["AAPL"],
        start_date="2024-01-01",
        get_latest=True,
    )

    assert captured["sql_file"] == "analyst_price_targets_latest.txt"
    assert "AAPL" in captured["filters"]["security_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
