from __future__ import annotations

import pandas as pd
import pytest

from data_loading.company_info_data_source import CompanyInfoDataSource
from data_loading.holdings_data_source import HoldingsDataSource
from data_loading.index_data_source import IndexDataSource
from data_loading.prices_data_source import PricesDataSource


def test_holdings_data_source_run_returns_transformed_long(monkeypatch):
    source_df = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-01"],
            "INDEX": ["S&P 500", "S&P 500"],
            "TICKER": ["aapl", "msft"],
            "WEIGHT": [0.5, 0.5],
        }
    )
    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        lambda **_: source_df,
    )

    source = HoldingsDataSource(source="index", portfolio="S&P 500")
    transformed = source.run()

    assert list(transformed.columns) == ["DATE", "INDEX", "TICKER", "WEIGHT"]
    assert transformed["TICKER"].tolist() == ["AAPL", "MSFT"]
    assert source.tickers == ["AAPL", "MSFT"]
    assert pd.api.types.is_datetime64_any_dtype(transformed["DATE"])


def test_holdings_data_source_passes_end_date(monkeypatch):
    captured: dict[str, object] = {}

    def fake_get_index_holdings(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"DATE": [], "TICKER": [], "WEIGHT": []})

    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        fake_get_index_holdings,
    )

    _ = HoldingsDataSource(
        source="index",
        portfolio="S&P 500",
        start_date="2026-01-01",
        end_date="2026-01-31",
    ).run()

    assert captured["start_date"] == "2026-01-01"
    assert captured["end_date"] == "2026-01-31"


def test_holdings_data_source_llm_source_uses_llm_loader(monkeypatch):
    captured: dict[str, object] = {}

    def fake_get_llm_holdings(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "DATE": ["2026-03-01"],
                "strategy": ["LLM1"],
                "TICKER": ["AAPL"],
                "WEIGHT": [0.1],
            }
        )

    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_llm_holdings",
        fake_get_llm_holdings,
    )

    transformed = HoldingsDataSource(
        source="llm",
        portfolio=["LLM1"],
        start_date="2026-01-01",
        end_date="2026-01-31",
    ).run()

    assert len(transformed) == 1
    assert transformed["TICKER"].iloc[0] == "AAPL"
    assert captured["llms"] == ["LLM1"]
    assert captured["start_date"] == "2026-01-01"
    assert captured["end_date"] == "2026-01-31"


def test_holdings_data_source_validation_requires_portfolio():
    import pytest

    with pytest.raises(ValueError, match="portfolio must be provided"):
        HoldingsDataSource(source="index", portfolio=None)


def test_prices_data_source_run_returns_transformed_long(monkeypatch):
    prices_long = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-01", "2026-03-02", "2026-03-02"],
            "TICKER": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "ADJ_CLOSE": [100.0, 200.0, 101.0, 202.0],
        }
    )
    monkeypatch.setattr(
        "data_loading.prices_data_source.get_prices",
        lambda **_: prices_long,
    )

    transformed = PricesDataSource(tickers=["AAPL", "MSFT"]).run()

    assert list(transformed.columns) == ["DATE", "TICKER", "ADJ_CLOSE"]
    assert len(transformed) == 4
    assert pd.api.types.is_datetime64_any_dtype(transformed["DATE"])


def test_prices_data_source_logs_warning_for_missing_ticker(monkeypatch):
    warnings: list[str] = []

    monkeypatch.setattr(
        "data_loading.prices_data_source.get_prices",
        lambda **_: pd.DataFrame(
            {
                "DATE": ["2026-03-01"],
                "TICKER": ["AAPL"],
                "ADJ_CLOSE": [100.0],
            }
        ),
    )

    def fake_log(text, type="info", **kwargs):  # noqa: ANN001
        if type == "warning":
            warnings.append(text)

    monkeypatch.setattr("data_loading.prices_data_source.log", fake_log)

    _ = PricesDataSource(tickers=["AAPL", "MSFT"]).run()
    assert any("MSFT" in msg for msg in warnings)


def test_prices_data_source_passes_end_date(monkeypatch):
    captured: dict[str, object] = {}

    def fake_get_prices(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(index=pd.to_datetime([]))

    monkeypatch.setattr(
        "data_loading.prices_data_source.get_prices",
        fake_get_prices,
    )

    _ = PricesDataSource(
        tickers=["AAPL"],
        start_date="2026-01-01",
        end_date="2026-01-31",
    ).run()

    assert captured["start_date"] == "2025-01-01"
    assert captured["end_date"] == "2026-01-31"


def test_prices_data_source_format_fills_internal_gaps():
    source = PricesDataSource(tickers=["AAPL", "MSFT"])
    long_df = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-02", "2026-03-03"],
            "TICKER": ["AAPL", "AAPL", "MSFT"],
            "ADJ_CLOSE": [100.0, 101.0, 300.0],
        }
    )

    source.transformed_data = source.transform(long_df)
    source.format()
    wide = source.formatted_data["prices_wide"]

    assert pd.isna(wide.loc[pd.Timestamp("2026-03-01"), "MSFT"])
    assert pd.isna(wide.loc[pd.Timestamp("2026-03-02"), "MSFT"])
    assert wide.loc[pd.Timestamp("2026-03-03"), "MSFT"] == 100.0
    assert wide.loc[pd.Timestamp("2026-03-01"), "AAPL"] == 100.0
    assert wide.loc[pd.Timestamp("2026-03-02"), "AAPL"] == 101.0
    assert "prices_wide_full_history" in source.formatted_data


def test_prices_data_source_format_fills_only_single_trailing_missing_day():
    source = PricesDataSource(tickers=["AAPL", "EWG"])
    long_df = pd.DataFrame(
        {
            "DATE": [
                "2026-03-01",
                "2026-03-02",
                "2026-03-03",
                "2026-03-01",
                "2026-03-02",
            ],
            "TICKER": ["AAPL", "AAPL", "AAPL", "EWG", "EWG"],
            "ADJ_CLOSE": [100.0, 101.0, 102.0, 50.0, 51.0],
        }
    )
    source.transformed_data = source.transform(long_df)
    source.format()
    wide = source.formatted_data["prices_wide"]

    assert (
        wide.loc[pd.Timestamp("2026-03-03"), "EWG"]
        == wide.loc[pd.Timestamp("2026-03-02"), "EWG"]
    )

    long_df_two_day_gap = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-01"],
            "TICKER": ["AAPL", "AAPL", "AAPL", "EWG"],
            "ADJ_CLOSE": [100.0, 101.0, 102.0, 50.0],
        }
    )
    source.transformed_data = source.transform(long_df_two_day_gap)
    source.format()
    wide_two_day = source.formatted_data["prices_wide"]
    assert pd.isna(wide_two_day.loc[pd.Timestamp("2026-03-03"), "EWG"])
    last_valid_date = source.formatted_data["last_valid_date"]
    assert "EWG" in last_valid_date.index
    assert last_valid_date["EWG"] == pd.Timestamp("2026-03-01")


def test_prices_data_source_outputs_full_and_windowed_history():
    source = PricesDataSource(tickers=["AAPL"], start_date="2026-03-02")
    long_df = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-02", "2026-03-03"],
            "TICKER": ["AAPL", "AAPL", "AAPL"],
            "ADJ_CLOSE": [100.0, 101.0, 102.0],
        }
    )
    source.transformed_data = source.transform(long_df)
    source.format()
    full_wide = source.formatted_data["prices_wide_full_history"]
    window_wide = source.formatted_data["prices_wide"]

    assert list(full_wide.index) == list(
        pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"])
    )
    assert list(window_wide.index) == list(pd.to_datetime(["2026-03-02", "2026-03-03"]))
    assert window_wide.loc[pd.Timestamp("2026-03-02"), "AAPL"] == 100.0


def test_prices_data_source_plot_prices_builds_figure(monkeypatch):
    called = {"show": 0}

    def fake_show(self):  # noqa: ANN001
        called["show"] += 1

    monkeypatch.setattr("visualization.figure.Figure.show", fake_show)

    source = PricesDataSource(tickers=["AAPL", "MSFT"], start_date="2026-03-01")
    long_df = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-01", "2026-03-02", "2026-03-02"],
            "TICKER": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "ADJ_CLOSE": [100.0, 200.0, 101.0, 202.0],
        }
    )
    source.transformed_data = source.transform(long_df)
    source.format()

    fig = source.plot_prices(tickers=["AAPL"], title="Prices")
    built = fig.build().fig
    assert built is not None
    assert len(built.data) == 1
    assert built.data[0].name == "AAPL"
    assert called["show"] == 1


def test_holdings_data_source_format_returns_long_and_wide(monkeypatch):
    source_df = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-01"],
            "INDEX": ["S&P 500", "S&P 500"],
            "TICKER": ["aapl", "msft"],
            "WEIGHT": [0.5, 0.5],
        }
    )
    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        lambda **_: source_df,
    )
    source = HoldingsDataSource(source="index", portfolio="S&P 500")
    _ = source.run()
    source.format()
    formatted = source.formatted_data

    assert "holdings_long" in formatted
    assert "weights_wide" in formatted
    assert "in_portfolio_wide" in formatted
    assert set(formatted["weights_wide"].columns) == {"AAPL", "MSFT"}
    in_portfolio = formatted["in_portfolio_wide"]
    assert bool(in_portfolio.loc[pd.Timestamp("2026-03-01"), "AAPL"]) is True
    assert bool(in_portfolio.loc[pd.Timestamp("2026-03-01"), "MSFT"]) is True


def test_holdings_data_source_in_portfolio_wide_marks_zero_weight_as_false(monkeypatch):
    source_df = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-01"],
            "INDEX": ["S&P 500", "S&P 500"],
            "TICKER": ["aapl", "msft"],
            "WEIGHT": [0.5, 0.0],
        }
    )
    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        lambda **_: source_df,
    )
    source = HoldingsDataSource(source="index", portfolio="S&P 500")
    _ = source.run()
    source.format()

    in_portfolio = source.formatted_data["in_portfolio_wide"]
    assert bool(in_portfolio.loc[pd.Timestamp("2026-03-01"), "AAPL"]) is True
    assert bool(in_portfolio.loc[pd.Timestamp("2026-03-01"), "MSFT"]) is False


def test_holdings_data_source_format_reindexes_to_prices_dates(monkeypatch):
    source_df = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-01"],
            "TICKER": ["aapl", "msft"],
            "WEIGHT": [0.5, 0.5],
        }
    )
    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        lambda **_: source_df,
    )
    source = HoldingsDataSource(source="index", portfolio="S&P 500")
    _ = source.run()
    target_dates = pd.to_datetime(["2026-03-01", "2026-03-02"])
    source.format(dates=target_dates)

    weights = source.formatted_data["weights_wide"]
    in_portfolio = source.formatted_data["in_portfolio_wide"]
    assert list(weights.index) == list(target_dates)
    assert pd.isna(weights.loc[pd.Timestamp("2026-03-02"), "AAPL"])
    assert bool(in_portfolio.loc[pd.Timestamp("2026-03-02"), "AAPL"]) is False


def test_holdings_data_source_ffills_single_all_nan_gap_after_reindex(monkeypatch):
    source_df = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-03", "2026-03-01", "2026-03-03"],
            "TICKER": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "WEIGHT": [0.6, 0.7, 0.4, 0.3],
        }
    )
    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        lambda **_: source_df,
    )
    source = HoldingsDataSource(source="index", portfolio="S&P 500")
    _ = source.run()
    target_dates = pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"])
    source.format(dates=target_dates)

    weights = source.formatted_data["weights_wide"]
    in_portfolio = source.formatted_data["in_portfolio_wide"]
    assert weights.loc[pd.Timestamp("2026-03-02"), "AAPL"] == 0.6
    assert weights.loc[pd.Timestamp("2026-03-02"), "MSFT"] == 0.4
    assert bool(in_portfolio.loc[pd.Timestamp("2026-03-02"), "AAPL"]) is True
    assert bool(in_portfolio.loc[pd.Timestamp("2026-03-02"), "MSFT"]) is True


def test_company_info_data_source_run_returns_transformed_info(monkeypatch):
    info = pd.DataFrame({"DATE": ["2026-03-01"], "TICKER": ["aapl"], "NAME": ["Apple"]})
    monkeypatch.setattr(
        "data_loading.company_info_data_source.get_company_info",
        lambda **_: info,
    )

    transformed = CompanyInfoDataSource(tickers=["AAPL"]).run()

    assert list(transformed.columns) == ["DATE", "TICKER", "NAME"]
    assert transformed["TICKER"].iloc[0] == "AAPL"


def test_company_info_data_source_logs_warning_for_missing_ticker(monkeypatch):
    warnings: list[str] = []

    monkeypatch.setattr(
        "data_loading.company_info_data_source.get_company_info",
        lambda **_: pd.DataFrame({"DATE": ["2026-03-01"], "TICKER": ["AAPL"]}),
    )

    def fake_log(text, type="info", **kwargs):  # noqa: ANN001
        if type == "warning":
            warnings.append(text)

    monkeypatch.setattr("data_loading.company_info_data_source.log", fake_log)

    _ = CompanyInfoDataSource(tickers=["AAPL", "MSFT"]).run()
    assert any("MSFT" in msg for msg in warnings)


def test_company_info_data_source_passes_end_date(monkeypatch):
    captured_info: dict[str, object] = {}

    def fake_get_company_info(**kwargs):
        captured_info.update(kwargs)
        return pd.DataFrame({"DATE": [], "TICKER": [], "NAME": []})

    monkeypatch.setattr(
        "data_loading.company_info_data_source.get_company_info",
        fake_get_company_info,
    )

    _ = CompanyInfoDataSource(
        tickers=["AAPL"],
        start_date="2026-01-01",
        end_date="2026-01-31",
    ).run()

    assert captured_info["start_date"] == "2026-01-01"
    assert captured_info["end_date"] == "2026-01-31"


def test_company_info_data_source_format_returns_company_info_payload(monkeypatch):
    info = pd.DataFrame(
        {
            "DATE": ["2026-03-01"],
            "TICKER": ["aapl"],
            "NAME": ["Apple"],
            "SECTOR": ["Technology"],
        }
    )
    monkeypatch.setattr(
        "data_loading.company_info_data_source.get_company_info",
        lambda **_: info,
    )

    source = CompanyInfoDataSource(tickers=["AAPL"])
    _ = source.run()
    source.format()
    formatted = source.formatted_data

    assert "company_info" in formatted
    assert "officers" not in formatted
    assert "sector_wide" in formatted
    assert formatted["company_info"]["TICKER"].iloc[0] == "AAPL"


def test_company_info_data_source_sector_wide_reindexes_and_ffills(monkeypatch):
    info = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-03"],
            "TICKER": ["aapl", "aapl"],
            "SECTOR": ["Technology", "Technology"],
        }
    )
    monkeypatch.setattr(
        "data_loading.company_info_data_source.get_company_info",
        lambda **_: info,
    )

    source = CompanyInfoDataSource(tickers=["AAPL"])
    _ = source.run()
    source.format(
        dates=pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]),
    )
    sector_wide = source.formatted_data["sector_wide"]

    assert list(sector_wide.index) == list(
        pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"])
    )
    assert sector_wide.loc[pd.Timestamp("2026-03-02"), "AAPL"] == "Technology"


def test_index_data_source_formats_returns(monkeypatch):
    prices_long = pd.DataFrame(
        {
            "DATE": ["2026-03-01", "2026-03-02", "2026-03-01", "2026-03-02"],
            "TICKER": ["^GSPC", "^GSPC", "^NDX", "^NDX"],
            "ADJ_CLOSE": [5000.0, 5100.0, 18000.0, 18180.0],
        }
    )
    monkeypatch.setattr(
        "data_loading.prices_data_source.get_prices",
        lambda **_: prices_long,
    )

    source = IndexDataSource(tickers=["^GSPC", "^NDX"])
    _ = source.run()
    source.format(dates=pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]))
    formatted = source.formatted_data

    assert "index_returns_long" in formatted
    assert "index_prices_wide" in formatted
    assert list(formatted["index_prices_wide"].index) == list(
        pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"])
    )
    returns_long = formatted["index_returns_long"]
    gspc = returns_long[returns_long["TICKER"] == "^GSPC"].sort_values("DATE")
    assert pd.isna(gspc["RETURN"].iloc[0])
    assert gspc["RETURN"].iloc[1] == pytest.approx(0.02)
