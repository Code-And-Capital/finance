import pandas as pd
import pytest

from handyman.insider_transactions import get_insider_transactions


def test_get_insider_transactions_reads_table_and_converts_dates(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.insider_transactions.read_table_by_filters",
        lambda **_: pd.DataFrame(
            {
                "TICKER": ["AAPL"],
                "DATE": ["2026-03-01"],
                "START_DATE": ["2025-12-31"],
                "TRANSACTION": ["BUY"],
            }
        ),
    )

    out = get_insider_transactions(tickers=["AAPL"], start_date="2026-01-01")

    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["START_DATE"])


def test_get_insider_transactions_missing_date_columns_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.insider_transactions.read_table_by_filters",
        lambda **_: pd.DataFrame({"TICKER": ["AAPL"], "TRANSACTION": ["BUY"]}),
    )

    out = get_insider_transactions(tickers=["AAPL"])

    assert len(out) == 1
    assert set(out.columns) == {"TICKER", "TRANSACTION"}


def test_get_insider_transactions_latest_uses_template_and_ignores_start_date(
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
                "START_DATE": ["2025-12-31"],
                "TRANSACTION": ["BUY"],
            }
        )

    monkeypatch.setattr(
        "handyman.insider_transactions.run_sql_template", fake_run_sql_template
    )

    out = get_insider_transactions(
        tickers=["AAPL"],
        start_date="2026-01-01",
        get_latest=True,
    )

    assert captured["sql_file"] == "insider_transactions_latest.txt"
    assert "AAPL" in captured["filters"]["security_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["START_DATE"])


def test_get_insider_transactions_latest_supports_figi_filter(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2026-03-01"]})

    monkeypatch.setattr(
        "handyman.insider_transactions.run_sql_template", fake_run_sql_template
    )

    _ = get_insider_transactions(figis=["BBG000B9XRY4"], get_latest=True)
    assert "FIGI" in captured["filters"]["security_filter"]
    assert "BBG000B9XRY4" in captured["filters"]["security_filter"]
