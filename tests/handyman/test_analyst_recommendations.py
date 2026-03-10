from __future__ import annotations

import pandas as pd
import pytest

from handyman.analyst_recommendations import (
    get_analyst_recommendations,
    get_analyst_upgrades_downgrades,
)


def test_get_analyst_recommendations_reads_table_and_converts_date(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.analyst_recommendations.read_table_by_filters",
        lambda **_: pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2026-03-01"], "VALUE": [1.0]}
        ),
    )

    out = get_analyst_recommendations(tickers=["AAPL"], start_date="2026-01-01")

    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_analyst_recommendations_latest_uses_template_and_ignores_start_date(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2026-03-01"], "VALUE": [1.0]}
        )

    monkeypatch.setattr(
        "handyman.analyst_recommendations.run_sql_template", fake_run_sql_template
    )

    out = get_analyst_recommendations(
        tickers=["AAPL"],
        start_date="2026-01-01",
        get_latest=True,
    )

    assert captured["sql_file"] == "analyst_recommendations_latest.txt"
    assert "AAPL" in captured["filters"]["security_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_analyst_recommendations_latest_supports_figi_filter(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2026-03-01"]})

    monkeypatch.setattr(
        "handyman.analyst_recommendations.run_sql_template", fake_run_sql_template
    )

    _ = get_analyst_recommendations(figis=["BBG000B9XRY4"], get_latest=True)
    assert "FIGI" in captured["filters"]["security_filter"]
    assert "BBG000B9XRY4" in captured["filters"]["security_filter"]


def test_get_analyst_upgrades_downgrades_reads_table_and_converts_dates(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.analyst_recommendations.read_table_by_filters",
        lambda **_: pd.DataFrame(
            {
                "TICKER": ["AAPL"],
                "DATE": ["2026-03-01"],
                "GRADEDATE": ["2026-02-15"],
                "VALUE": [1.0],
            }
        ),
    )

    out = get_analyst_upgrades_downgrades(tickers=["AAPL"], start_date="2026-01-01")

    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["GRADEDATE"])


def test_get_analyst_upgrades_downgrades_latest_uses_template_and_ignores_start_date(
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
                "GRADEDATE": ["2026-02-15"],
                "VALUE": [1.0],
            }
        )

    monkeypatch.setattr(
        "handyman.analyst_recommendations.run_sql_template", fake_run_sql_template
    )

    out = get_analyst_upgrades_downgrades(
        tickers=["AAPL"],
        start_date="2026-01-01",
        get_latest=True,
    )

    assert captured["sql_file"] == "analyst_upgrades_downgrades_latest.txt"
    assert "AAPL" in captured["filters"]["security_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["GRADEDATE"])
