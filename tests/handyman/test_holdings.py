from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from handyman.holdings import (
    get_index_holdings,
    get_llm_holdings,
)
from handyman.holders import get_institutional_holders, get_major_holders


@pytest.fixture
def sample_holdings_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-02"],
            "INDEX": ["SP500", "NASDAQ100"],
            "TICKER": ["AAPL", "MSFT"],
            "WEIGHT": [0.05, 0.04],
        }
    )


@pytest.fixture
def sample_llm_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-02"],
            "strategy": ["LLM1", "LLM2"],
            "TICKER": ["AAPL", "MSFT"],
        }
    )


def test_get_index_holdings_builds_all_filters(
    monkeypatch: pytest.MonkeyPatch, sample_holdings_df: pd.DataFrame
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return sample_holdings_df

    monkeypatch.setattr("handyman.holdings.run_sql_template", fake_run_sql_template)

    out = get_index_holdings(
        indices=np.array(["SP500", "NASDAQ100"]),
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-01",
    )

    assert captured["sql_file"] == "holdings.txt"
    filters = captured["filters"]
    assert "SP500" in filters["index_filter"]
    assert "NASDAQ100" in filters["index_filter"]
    assert '"INDEX"' in filters["index_filter"]
    assert "AAPL" in filters["ticker_filter"]
    assert "MSFT" in filters["ticker_filter"]
    assert filters["date_filter"] == "AND DATE >= '2024-01-01'"
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_index_holdings_includes_end_date_filter(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return pd.DataFrame(
            {"DATE": ["2024-01-01"], "TICKER": ["AAPL"], "WEIGHT": [0.1]}
        )

    monkeypatch.setattr("handyman.holdings.run_sql_template", fake_run_sql_template)

    _ = get_index_holdings(start_date="2024-01-01", end_date="2024-01-31")

    assert "DATE >= '2024-01-01'" in captured["filters"]["date_filter"]
    assert "DATE <= '2024-01-31'" in captured["filters"]["date_filter"]


def test_get_index_holdings_without_filters(
    monkeypatch: pytest.MonkeyPatch, sample_holdings_df: pd.DataFrame
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return sample_holdings_df

    monkeypatch.setattr("handyman.holdings.run_sql_template", fake_run_sql_template)

    out = get_index_holdings()

    assert captured["filters"] == {
        "index_filter": "",
        "ticker_filter": "",
        "date_filter": "",
    }
    assert len(out) == 2


def test_get_index_holdings_latest_uses_latest_template_and_ignores_start_date(
    monkeypatch: pytest.MonkeyPatch,
    sample_holdings_df: pd.DataFrame,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return sample_holdings_df

    monkeypatch.setattr("handyman.holdings.run_sql_template", fake_run_sql_template)

    out = get_index_holdings(
        indices=["SP500"], start_date="2024-01-01", get_latest=True
    )

    assert captured["sql_file"] == "holdings_latest.txt"
    assert "SP500" in captured["filters"]["index_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert len(out) == 2


@pytest.mark.parametrize(
    "kwargs", [{"indices": 123}, {"tickers": 123}, {"indices": ["SP500", 1]}]
)
def test_get_index_holdings_invalid_inputs_raise(kwargs):
    with pytest.raises(TypeError):
        get_index_holdings(**kwargs)


def test_get_index_holdings_missing_date_column_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "handyman.holdings.run_sql_template",
        lambda **_: pd.DataFrame({"TICKER": ["AAPL"]}),
    )

    with pytest.raises(ValueError, match="Expected column 'DATE'"):
        get_index_holdings()


def test_get_llm_holdings_builds_filters(
    monkeypatch: pytest.MonkeyPatch, sample_llm_df: pd.DataFrame
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return sample_llm_df

    monkeypatch.setattr("handyman.holdings.run_sql_template", fake_run_sql_template)

    out = get_llm_holdings(llms=np.array(["LLM1", "LLM3"]), start_date="2024-01-01")

    assert captured["sql_file"] == "llm_holdings.txt"
    filters = captured["filters"]
    assert "LLM1" in filters["llm_filter"]
    assert "LLM3" in filters["llm_filter"]
    assert filters["date_filter"] == "AND DATE >= '2024-01-01'"
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_llm_holdings_includes_end_date_filter(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return pd.DataFrame(
            {"DATE": ["2024-01-01"], "strategy": ["LLM1"], "TICKER": ["AAPL"]}
        )

    monkeypatch.setattr("handyman.holdings.run_sql_template", fake_run_sql_template)

    _ = get_llm_holdings(start_date="2024-01-01", end_date="2024-01-31")

    assert "DATE >= '2024-01-01'" in captured["filters"]["date_filter"]
    assert "DATE <= '2024-01-31'" in captured["filters"]["date_filter"]


def test_get_llm_holdings_no_filters(
    monkeypatch: pytest.MonkeyPatch, sample_llm_df: pd.DataFrame
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return sample_llm_df

    monkeypatch.setattr("handyman.holdings.run_sql_template", fake_run_sql_template)

    out = get_llm_holdings()

    assert captured["filters"] == {"llm_filter": "", "date_filter": ""}
    assert len(out) == 2


def test_get_llm_holdings_latest_uses_latest_template_and_ignores_start_date(
    monkeypatch: pytest.MonkeyPatch,
    sample_llm_df: pd.DataFrame,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return sample_llm_df

    monkeypatch.setattr("handyman.holdings.run_sql_template", fake_run_sql_template)

    out = get_llm_holdings(llms=["LLM1"], start_date="2024-01-01", get_latest=True)

    assert captured["sql_file"] == "llm_holdings_latest.txt"
    assert "LLM1" in captured["filters"]["llm_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert len(out) == 2


def test_get_llm_holdings_invalid_llm_type_raises():
    with pytest.raises(TypeError):
        get_llm_holdings(llms=["LLM1", 2])


def test_get_institutional_holders_reads_table_and_converts_dates(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.holders.read_table_by_filters",
        lambda **_: pd.DataFrame(
            {
                "TICKER": ["AAPL"],
                "DATE": ["2026-03-01"],
                "DATE_REPORTED": ["2025-12-31"],
                "SHARES": [1000],
            }
        ),
    )

    out = get_institutional_holders(tickers=["AAPL"], start_date="2026-01-01")

    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["DATE_REPORTED"])


def test_get_major_holders_reads_table_and_converts_date(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "handyman.holders.read_table_by_filters",
        lambda **_: pd.DataFrame(
            {
                "TICKER": ["AAPL"],
                "DATE": ["2026-03-01"],
                "VALUE": [0.25],
            }
        ),
    )

    out = get_major_holders(tickers=["AAPL"], start_date="2026-01-01")

    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_institutional_holders_latest_uses_template_and_ignores_start_date(
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
                "DATE_REPORTED": ["2025-12-31"],
            }
        )

    monkeypatch.setattr("handyman.holders.run_sql_template", fake_run_sql_template)

    out = get_institutional_holders(
        tickers=["AAPL"],
        start_date="2026-01-01",
        get_latest=True,
    )

    assert captured["sql_file"] == "institutional_holders_latest.txt"
    assert "AAPL" in captured["filters"]["ticker_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert pd.api.types.is_datetime64_any_dtype(out["DATE_REPORTED"])


def test_get_major_holders_latest_uses_template_and_ignores_start_date(
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
                "VALUE": [0.25],
            }
        )

    monkeypatch.setattr("handyman.holders.run_sql_template", fake_run_sql_template)

    out = get_major_holders(
        tickers=["AAPL"],
        start_date="2026-01-01",
        get_latest=True,
    )

    assert captured["sql_file"] == "major_holders_latest.txt"
    assert "AAPL" in captured["filters"]["ticker_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
