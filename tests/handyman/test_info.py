import numpy as np
import pandas as pd
import pytest

from handyman.company_info import get_company_info, get_officers


@pytest.fixture
def sample_company_info_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "DATE": ["2024-01-01", "2024-01-02"],
            "SECTOR": ["Tech", "Tech"],
        }
    )


def test_get_company_info_builds_filters_and_converts_dates(
    monkeypatch: pytest.MonkeyPatch, sample_company_info_df: pd.DataFrame
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        captured["configs_path"] = configs_path
        return sample_company_info_df

    monkeypatch.setattr("handyman.company_info.run_sql_template", fake_run_sql_template)

    out = get_company_info(
        tickers=np.array(["AAPL", "MSFT"]),
        start_date="2024-01-01",
        configs_path="cfg.yml",
    )

    assert captured["sql_file"] == "company_info.txt"
    filters = captured["filters"]
    assert "AAPL" in filters["security_filter"]
    assert "MSFT" in filters["security_filter"]
    assert filters["date_filter"] == "AND DATE >= '2024-01-01'"
    assert captured["configs_path"] == "cfg.yml"
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])


def test_get_company_info_supports_end_date_filter(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})

    monkeypatch.setattr("handyman.company_info.run_sql_template", fake_run_sql_template)

    _ = get_company_info(start_date="2024-01-01", end_date="2024-01-31")

    date_filter = captured["filters"]["date_filter"]
    assert "DATE >= '2024-01-01'" in date_filter
    assert "DATE <= '2024-01-31'" in date_filter


def test_get_company_info_without_filters(
    monkeypatch: pytest.MonkeyPatch, sample_company_info_df: pd.DataFrame
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return sample_company_info_df

    monkeypatch.setattr("handyman.company_info.run_sql_template", fake_run_sql_template)

    out = get_company_info()

    assert captured["filters"] == {"security_filter": "", "date_filter": ""}
    assert len(out) == 2


def test_get_company_info_supports_figi_filter(
    monkeypatch: pytest.MonkeyPatch, sample_company_info_df: pd.DataFrame
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["filters"] = filters
        return sample_company_info_df

    monkeypatch.setattr("handyman.company_info.run_sql_template", fake_run_sql_template)

    _ = get_company_info(figis=["BBG000B9XRY4"])
    assert "FIGI" in captured["filters"]["security_filter"]
    assert "BBG000B9XRY4" in captured["filters"]["security_filter"]


def test_get_company_info_rejects_tickers_and_figis_together():
    with pytest.raises(ValueError, match="Provide only one of `tickers` or `figis`"):
        get_company_info(tickers=["AAPL"], figis=["BBG000B9XRY4"])


def test_get_company_info_latest_uses_latest_template_and_drops_helper_column(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return pd.DataFrame(
            {
                "TICKER": ["AAPL"],
                "DATE": ["2024-01-02"],
                "SECTOR": ["Tech"],
                "_ROW_NUM": [1],
            }
        )

    monkeypatch.setattr("handyman.company_info.run_sql_template", fake_run_sql_template)

    out = get_company_info(get_latest=True)

    assert captured["sql_file"] == "company_info_latest.txt"
    assert "_ROW_NUM" not in out.columns
    assert len(out) == 1


def test_get_company_info_latest_ignores_start_date_filter(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]})

    monkeypatch.setattr("handyman.company_info.run_sql_template", fake_run_sql_template)

    _ = get_company_info(get_latest=True, start_date="2024-01-01")

    assert captured["sql_file"] == "company_info_latest.txt"
    assert captured["filters"]["date_filter"] == ""


@pytest.mark.parametrize("bad_tickers", [123, ["AAPL", 1]])
def test_get_company_info_invalid_tickers_raise(bad_tickers):
    with pytest.raises(TypeError):
        get_company_info(tickers=bad_tickers)


def test_get_company_info_empty_ticker_raises():
    with pytest.raises(ValueError, match="must not contain empty strings"):
        get_company_info(tickers=["AAPL", " "])


def test_get_company_info_invalid_start_date_raises():
    with pytest.raises(ValueError, match="start_date must be parseable"):
        get_company_info(start_date="not-a-date")


def test_get_company_info_missing_date_column_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "handyman.company_info.run_sql_template",
        lambda **_: pd.DataFrame({"TICKER": ["AAPL"]}),
    )

    with pytest.raises(ValueError, match="Expected column 'DATE'"):
        get_company_info()


def test_get_officers_reads_table_and_converts_date(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_read_table_by_filters(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-01"]})

    monkeypatch.setattr(
        "handyman.company_info.read_table_by_filters", fake_read_table_by_filters
    )
    out = get_officers(tickers=["AAPL"], end_date="2024-12-31")
    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert captured["end_date"] == "2024-12-31"


def test_get_officers_latest_uses_latest_template_and_ignores_start_date(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_run_sql_template(*, sql_file, filters, configs_path):
        captured["sql_file"] = sql_file
        captured["filters"] = filters
        return pd.DataFrame(
            {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "NAME": ["Tim Cook"]}
        )

    monkeypatch.setattr("handyman.company_info.run_sql_template", fake_run_sql_template)

    out = get_officers(tickers=["AAPL"], start_date="2024-01-01", get_latest=True)

    assert captured["sql_file"] == "officers_latest.txt"
    assert "AAPL" in captured["filters"]["security_filter"]
    assert captured["filters"]["date_filter"] == ""
    assert len(out) == 1
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
