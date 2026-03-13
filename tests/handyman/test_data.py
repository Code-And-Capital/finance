import pandas as pd
import pytest

from handyman.fundamentals import (
    get_earnings_surprises,
    get_eps_estimates,
    get_eps_revisions,
    get_fundamentals,
    get_growth_estimates,
    get_revenue_estimates,
)
from handyman.options import get_options


def test_get_fundamentals_invalid_combo_raises():
    with pytest.raises(ValueError, match="Invalid fundamentals selection"):
        get_fundamentals(statement_type="invalid", annual=True)


def test_get_fundamentals_latest_uses_latest_template(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-01"]})

    monkeypatch.setattr("handyman.fundamentals.run_sql_template", fake_run_sql_template)
    monkeypatch.setattr(
        "handyman.fundamentals.read_table_by_filters", lambda **_: pd.DataFrame()
    )

    out = get_fundamentals(
        statement_type="financial",
        annual=True,
        tickers=["AAPL"],
        start_date="2024-01-01",
        get_latest=True,
    )

    assert captured["sql_file"] == "fundamentals_latest.txt"
    assert "AAPL" in captured["filters"]["security_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert captured["filters"]["table_name"] == "[financial_annual]"
    assert captured["filters"]["partition_column"] == "[REPORT_DATE]"
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_options_reads_table_and_converts_dates(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "handyman.options.read_table_by_filters",
        lambda **_: pd.DataFrame(
            {
                "TICKER": ["AAPL"],
                "DATE": ["2026-03-01"],
                "LASTTRADEDATE": ["2026-03-01 15:30:00"],
                "EXPIRATION": ["2026-06-19"],
            }
        ),
    )

    out = get_options(tickers=["AAPL"])

    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["LASTTRADEDATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["EXPIRATION"])


def test_get_options_latest_uses_template_and_ignores_start_date(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return pd.DataFrame(
            {
                "TICKER": ["AAPL"],
                "DATE": ["2026-03-01"],
                "LASTTRADEDATE": ["2026-03-01 15:30:00"],
                "EXPIRATION": ["2026-06-19"],
                "_ROW_NUM": [1],
            }
        )

    monkeypatch.setattr("handyman.options.run_sql_template", fake_run_sql_template)

    out = get_options(tickers=["AAPL"], start_date="2026-01-01", get_latest=True)

    assert captured["sql_file"] == "options_latest.txt"
    assert "AAPL" in captured["filters"]["security_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert "_ROW_NUM" not in out.columns


def test_get_eps_revisions_reads_table_and_converts_date(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.fundamentals.read_table_by_filters",
        lambda **_: pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2026-03-01"], "VALUE": [1.0]}
        ),
    )

    out = get_eps_revisions(tickers=["AAPL"], start_date="2026-01-01")

    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_eps_revisions_latest_uses_template_and_ignores_start_date(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2026-03-01"], "VALUE": [1.0]}
        )

    monkeypatch.setattr("handyman.fundamentals.run_sql_template", fake_run_sql_template)

    out = get_eps_revisions(tickers=["AAPL"], start_date="2026-01-01", get_latest=True)

    assert captured["sql_file"] == "eps_revisions_latest.txt"
    assert "AAPL" in captured["filters"]["security_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_earnings_surprises_reads_table_and_converts_dates(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.fundamentals.read_table_by_filters",
        lambda **_: pd.DataFrame(
            {
                "TICKER": ["AAPL"],
                "DATE": ["2026-03-01"],
                "EARNINGS_DATE": ["2025-12-31"],
                "VALUE": [1.0],
            }
        ),
    )

    out = get_earnings_surprises(tickers=["AAPL"], start_date="2026-01-01")

    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["EARNINGS_DATE"])


def test_get_earnings_surprises_latest_uses_template_and_ignores_start_date(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return pd.DataFrame(
            {
                "TICKER": ["AAPL"],
                "DATE": ["2026-03-01"],
                "EARNINGS_DATE": ["2025-12-31"],
                "VALUE": [1.0],
            }
        )

    monkeypatch.setattr("handyman.fundamentals.run_sql_template", fake_run_sql_template)

    out = get_earnings_surprises(
        tickers=["AAPL"],
        start_date="2026-01-01",
        get_latest=True,
    )

    assert captured["sql_file"] == "earnings_surprises_latest.txt"
    assert "AAPL" in captured["filters"]["security_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["EARNINGS_DATE"])


def test_get_eps_estimates_reads_table_and_converts_date(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.fundamentals.read_table_by_filters",
        lambda **_: pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2026-03-01"], "VALUE": [1.0]}
        ),
    )

    out = get_eps_estimates(tickers=["AAPL"], start_date="2026-01-01")

    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_revenue_estimates_reads_table_and_converts_date(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.fundamentals.read_table_by_filters",
        lambda **_: pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2026-03-01"], "VALUE": [2.0]}
        ),
    )

    out = get_revenue_estimates(tickers=["AAPL"], start_date="2026-01-01")

    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_growth_estimates_reads_table_and_converts_date(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.fundamentals.read_table_by_filters",
        lambda **_: pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2026-03-01"], "VALUE": [3.0]}
        ),
    )

    out = get_growth_estimates(tickers=["AAPL"], start_date="2026-01-01")

    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_eps_estimates_latest_uses_fundamentals_latest_template(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2026-03-01"], "VALUE": [1.0]}
        )

    monkeypatch.setattr("handyman.fundamentals.run_sql_template", fake_run_sql_template)

    out = get_eps_estimates(tickers=["AAPL"], start_date="2026-01-01", get_latest=True)

    assert captured["sql_file"] == "fundamentals_latest.txt"
    assert "AAPL" in captured["filters"]["security_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert captured["filters"]["table_name"] == "[eps_estimates]"
    assert captured["filters"]["partition_column"] == "[PERIOD]"
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_options_latest_supports_figi_filter(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2026-03-01"]})

    monkeypatch.setattr("handyman.options.run_sql_template", fake_run_sql_template)

    _ = get_options(figis=["BBG000B9XRY4"], get_latest=True)
    assert "FIGI" in captured["filters"]["security_filter"]
    assert "BBG000B9XRY4" in captured["filters"]["security_filter"]


def test_get_fundamentals_latest_supports_figi_filter(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-01"]})

    monkeypatch.setattr("handyman.fundamentals.run_sql_template", fake_run_sql_template)

    _ = get_fundamentals(
        statement_type="financial",
        annual=True,
        figis=["BBG000B9XRY4"],
        get_latest=True,
    )
    assert "FIGI" in captured["filters"]["security_filter"]
    assert "BBG000B9XRY4" in captured["filters"]["security_filter"]
