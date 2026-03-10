from __future__ import annotations

import pandas as pd

import scripts.daily_market_data as daily_script


def test_build_ticker_to_figi_map_normalizes_and_deduplicates():
    holdings = pd.DataFrame(
        {
            "TICKER": [" aapl ", "AAPL", "msft"],
            "FIGI": ["FIGI_AAPL", "FIGI_AAPL_OLD", "FIGI_MSFT"],
        }
    )

    out = daily_script._build_ticker_to_figi_map(holdings)

    assert out == {"AAPL": "FIGI_AAPL", "MSFT": "FIGI_MSFT"}


def test_run_daily_market_data_passes_figi_mapping(monkeypatch):
    monkeypatch.setattr(
        daily_script,
        "ETF_URLS",
        {"S&P 500": "http://example.test"},
    )

    class DummyDownloadHoldings:
        def __init__(self, fund_name, url):
            self.fund_name = fund_name
            self.url = url

        def run(self, **_kwargs):
            return pd.DataFrame(
                {
                    "TICKER": ["AAPL", "MSFT"],
                    "FIGI": ["FIGI_AAPL", "FIGI_MSFT"],
                }
            )

    class DummyPricingData:
        calls: list[dict] = []

        def __init__(self, tickers):
            self.tickers = tickers

        def run(self, **kwargs):
            self.__class__.calls.append(
                {"tickers": list(self.tickers), "kwargs": kwargs}
            )
            return pd.DataFrame({"TICKER": list(self.tickers)})

    class DummyInfoData:
        def __init__(self, tickers):
            self.tickers = tickers

        def run(self, **_kwargs):
            base = pd.DataFrame({"TICKER": list(self.tickers)})
            return base.copy(), base.copy()

    class DummySingleFrame:
        def __init__(self, tickers):
            self.tickers = tickers

        def run(self, **_kwargs):
            return pd.DataFrame({"TICKER": list(self.tickers)})

    class DummyFinancialData(DummySingleFrame):
        def run(self, **_kwargs):
            return {"financial_annual": pd.DataFrame({"TICKER": list(self.tickers)})}

    class DummyEstimatesData(DummySingleFrame):
        def run(self, **_kwargs):
            frame = pd.DataFrame({"TICKER": list(self.tickers)})
            return {"eps": frame, "revenue": frame, "growth": frame}

    monkeypatch.setattr(daily_script, "DownloadHoldings", DummyDownloadHoldings)
    monkeypatch.setattr(daily_script, "PricingData", DummyPricingData)
    monkeypatch.setattr(daily_script, "InfoData", DummyInfoData)
    monkeypatch.setattr(daily_script, "AnalystPriceTargetsData", DummySingleFrame)
    monkeypatch.setattr(daily_script, "AnalystRecommendationsData", DummySingleFrame)
    monkeypatch.setattr(daily_script, "AnalystUpgradesDowngradesData", DummySingleFrame)
    monkeypatch.setattr(daily_script, "InstitutionalHolders", DummySingleFrame)
    monkeypatch.setattr(daily_script, "MajorHolders", DummySingleFrame)
    monkeypatch.setattr(daily_script, "InsiderTransactionsData", DummySingleFrame)
    monkeypatch.setattr(daily_script, "FinancialData", DummyFinancialData)
    monkeypatch.setattr(daily_script, "EPSRevisionsData", DummySingleFrame)
    monkeypatch.setattr(daily_script, "EstimatesData", DummyEstimatesData)
    monkeypatch.setattr(daily_script, "OptionsData", DummySingleFrame)

    output = daily_script.run_daily_market_data(
        write_to_db=False,
        include_options=False,
        indices={"^GSPC": "FIGI_SPX"},
    )

    assert output["tickers_run"]["holdings_tickers"] == ["AAPL", "MSFT"]
    assert len(DummyPricingData.calls) == 2
    assert DummyPricingData.calls[0]["kwargs"]["ticker_to_figi"] == {
        "^GSPC": "FIGI_SPX"
    }
    assert DummyPricingData.calls[1]["kwargs"]["ticker_to_figi"] == {
        "AAPL": "FIGI_AAPL",
        "MSFT": "FIGI_MSFT",
    }
