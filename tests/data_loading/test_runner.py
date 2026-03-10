from __future__ import annotations

import pandas as pd

from data_loading.company_info_data_source import CompanyInfoDataSource
from data_loading.holdings_data_source import HoldingsDataSource
from data_loading.index_data_source import IndexDataSource
from data_loading.prices_data_source import PricesDataSource
from data_loading.runner import Runner


def test_runner_executes_and_propagates_figis(monkeypatch):
    captured: dict[str, object] = {"price_calls": []}

    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        lambda **_: pd.DataFrame(
            {
                "DATE": ["2026-03-01", "2026-03-01", "2026-03-01"],
                "FIGI": ["aapl", "msft", "aapl"],
                "WEIGHT": [0.5, 0.5, 0.5],
            }
        ),
    )

    def fake_get_prices(**kwargs):
        captured["price_calls"].append(kwargs["figis"])
        captured["prices_figis"] = kwargs["figis"]
        captured["prices_start_date"] = kwargs.get("start_date")
        captured["prices_end_date"] = kwargs.get("end_date")
        captured["prices_configs_path"] = kwargs.get("configs_path")
        figis = kwargs["figis"]
        if figis == ["AAPL", "MSFT"]:
            return pd.DataFrame(
                {
                    "DATE": ["2026-03-01", "2026-03-01"],
                    "FIGI": ["AAPL", "MSFT"],
                    "ADJ_CLOSE": [100.0, 200.0],
                }
            )
        return pd.DataFrame(
            {
                "DATE": ["2026-03-01", "2026-03-02"],
                "FIGI": ["^GSPC", "^GSPC"],
                "ADJ_CLOSE": [5000.0, 5050.0],
            }
        )

    def fake_get_company_info(**kwargs):
        captured["security_figis"] = kwargs["figis"]
        captured["security_start_date"] = kwargs.get("start_date")
        captured["security_end_date"] = kwargs.get("end_date")
        captured["security_configs_path"] = kwargs.get("configs_path")
        return pd.DataFrame(
            {
                "DATE": ["2026-03-01", "2026-03-01"],
                "FIGI": ["AAPL", "MSFT"],
                "NAME": ["Apple", "Microsoft"],
                "SECTOR": ["Technology", "Technology"],
            }
        )

    monkeypatch.setattr("data_loading.prices_data_source.get_prices", fake_get_prices)
    monkeypatch.setattr(
        "data_loading.company_info_data_source.get_company_info", fake_get_company_info
    )

    runner = Runner(
        portfolio="S&P 500",
        holdings_source="index",
        index_figis=["^GSPC"],
        start_date="2026-01-01",
        end_date="2026-02-28",
        configs_path="config/default.yaml",
    )
    out = runner.run()

    assert isinstance(out["holdings"], HoldingsDataSource)
    assert isinstance(out["prices"], PricesDataSource)
    assert isinstance(out["index"], IndexDataSource)
    assert isinstance(out["security"], CompanyInfoDataSource)
    assert captured["price_calls"][0] == ["AAPL", "MSFT"]
    assert captured["price_calls"][1] == ["^GSPC"]
    assert captured["security_figis"] == ["AAPL", "MSFT"]
    assert captured["prices_start_date"] == "2025-01-01"
    assert captured["prices_end_date"] == "2026-02-28"
    assert captured["prices_configs_path"] == "config/default.yaml"
    assert captured["security_start_date"] == "2026-01-01"
    assert captured["security_end_date"] == "2026-02-28"
    assert captured["security_configs_path"] == "config/default.yaml"
    prices_dates = out["prices"].formatted_data["prices_wide"].index
    holdings_dates = out["holdings"].formatted_data["weights_wide"].index
    index_dates = out["index"].formatted_data["index_prices_wide"].index
    security_dates = out["security"].formatted_data["sector_wide"].index
    assert list(holdings_dates) == list(prices_dates)
    assert list(index_dates) == list(prices_dates)
    assert list(security_dates) == list(prices_dates)


def test_runner_skips_index_when_index_figis_is_none(monkeypatch):
    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        lambda **_: pd.DataFrame(
            {
                "DATE": ["2026-03-01"],
                "FIGI": ["aapl"],
                "WEIGHT": [0.5],
            }
        ),
    )
    monkeypatch.setattr(
        "data_loading.prices_data_source.get_prices",
        lambda **_: pd.DataFrame(
            {
                "DATE": ["2026-03-01"],
                "FIGI": ["AAPL"],
                "ADJ_CLOSE": [100.0],
            }
        ),
    )
    monkeypatch.setattr(
        "data_loading.company_info_data_source.get_company_info",
        lambda **_: pd.DataFrame({"DATE": ["2026-03-01"], "FIGI": ["AAPL"]}),
    )

    runner = Runner(portfolio="S&P 500", index_figis=None)
    out = runner.run()
    assert out["index"] is None


def test_runner_raises_when_holdings_has_no_figi_column(monkeypatch):
    import pytest

    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        lambda **_: pd.DataFrame({"DATE": ["2026-03-01"], "WEIGHT": [1.0]}),
    )
    runner = Runner(portfolio="S&P 500")

    with pytest.raises(ValueError, match="FIGI"):
        runner.run()


def test_runner_raises_when_no_figis_available(monkeypatch):
    import pytest

    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        lambda **_: pd.DataFrame(
            {"DATE": ["2026-03-01"], "FIGI": [None], "WEIGHT": [1.0]}
        ),
    )
    runner = Runner(portfolio="S&P 500")

    with pytest.raises(ValueError, match="No FIGIs"):
        runner.run()
