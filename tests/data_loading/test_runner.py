from types import SimpleNamespace

import pandas as pd

from bt.core import Strategy
from data_loading.company_info_data_source import CompanyInfoDataSource
from data_loading.holdings_data_source import HoldingsDataSource
from data_loading.index_data_source import IndexDataSource
from data_loading.prices_data_source import PricesDataSource
from bt.runner import Runner


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
    out = runner.load_data()

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
    out = runner.load_data()
    assert out["index"] is None


def test_runner_raises_when_holdings_has_no_figi_column(monkeypatch):
    import pytest

    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        lambda **_: pd.DataFrame({"DATE": ["2026-03-01"], "WEIGHT": [1.0]}),
    )
    runner = Runner(portfolio="S&P 500")

    with pytest.raises(ValueError, match="FIGI"):
        runner.load_data()


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
        runner.load_data()


def test_runner_run_backtest_passes_live_start_date(monkeypatch):
    captured: list[dict[str, object]] = []
    summary_captured: dict[str, object] = {}

    full_history = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    live_window = full_history.iloc[1:]

    class FakeBacktest:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.name = kwargs["strategy"].name

        def run(self):
            return None

    monkeypatch.setattr("bt.runner.Backtest", FakeBacktest)
    monkeypatch.setattr(
        Runner,
        "run_strategies",
        staticmethod(
            lambda *backtests, benchmark=None, figi_to_ticker=None, progress_bar=True: summary_captured.update(
                {
                    "benchmark": benchmark,
                    "figi_to_ticker": figi_to_ticker,
                    "progress_bar": progress_bar,
                }
            )
            or list(backtests)
        ),
    )

    runner = Runner(portfolio="S&P 500")
    runner.prices_data_source = SimpleNamespace(
        formatted_data={
            "prices_wide_full_history": full_history,
            "prices_wide": live_window,
        }
    )
    runner.holdings_data_source = SimpleNamespace(formatted_data={})
    runner.index_data_source = SimpleNamespace(
        formatted_data={
            "index_prices_wide": pd.DataFrame(
                {"FIGI_SPX": [100.0, 101.0]},
                index=live_window.index,
            )
        }
    )
    runner.figi_to_ticker = {"FIGI_SPX": "SPX"}
    runner.security_data_source = None

    result = runner.run_backtest(Strategy("demo"), progress_bar=False)

    assert len(result) == 1
    assert captured[0]["prices"] is full_history
    assert captured[0]["live_start_date"] == live_window.index[0]
    assert captured[0]["commissions"](1.0, 100.0) == 0.0
    assert captured[0]["integer_positions"] is False
    assert summary_captured["benchmark"].equals(
        pd.DataFrame({"SPX": [100.0, 101.0]}, index=live_window.index)
    )
    assert summary_captured["figi_to_ticker"] == {"FIGI_SPX": "SPX"}


def test_runner_run_backtest_forwards_custom_commissions(monkeypatch):
    captured: list[dict[str, object]] = []

    full_history = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    live_window = full_history.iloc[1:]

    class FakeBacktest:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.name = kwargs["strategy"].name

        def run(self):
            return None

    monkeypatch.setattr("bt.runner.Backtest", FakeBacktest)
    monkeypatch.setattr(
        Runner,
        "run_strategies",
        staticmethod(lambda *backtests, **kwargs: list(backtests)),
    )

    runner = Runner(portfolio="S&P 500")
    runner.prices_data_source = SimpleNamespace(
        formatted_data={
            "prices_wide_full_history": full_history,
            "prices_wide": live_window,
        }
    )
    runner.holdings_data_source = SimpleNamespace(formatted_data={})
    runner.index_data_source = None
    runner.security_data_source = None

    commission_fn = lambda quantity, price: 5.0  # noqa: E731
    runner.run_backtest(
        Strategy("demo"),
        progress_bar=False,
        commissions=commission_fn,
    )

    assert captured[0]["commissions"] is commission_fn


def test_runner_run_backtest_omits_benchmark_without_index_source(monkeypatch):
    captured: dict[str, object] = {}

    full_history = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    live_window = full_history.iloc[1:]

    class FakeBacktest:
        def __init__(self, **kwargs):
            self.name = kwargs["strategy"].name

        def run(self):
            return None

    monkeypatch.setattr("bt.runner.Backtest", FakeBacktest)
    monkeypatch.setattr(
        Runner,
        "run_strategies",
        staticmethod(
            lambda *backtests, benchmark=None, figi_to_ticker=None, progress_bar=True: captured.update(
                {"benchmark": benchmark, "figi_to_ticker": figi_to_ticker}
            )
            or list(backtests)
        ),
    )

    runner = Runner(portfolio="S&P 500")
    runner.prices_data_source = SimpleNamespace(
        formatted_data={
            "prices_wide_full_history": full_history,
            "prices_wide": live_window,
        }
    )
    runner.holdings_data_source = SimpleNamespace(formatted_data={})
    runner.index_data_source = None
    runner.security_data_source = None

    runner.run_backtest(Strategy("demo"), progress_bar=False)

    assert captured["benchmark"] is None
    assert captured["figi_to_ticker"] == {}


def test_runner_builds_figi_ticker_maps(monkeypatch):
    monkeypatch.setattr(
        "data_loading.holdings_data_source.get_index_holdings",
        lambda **_: pd.DataFrame(
            {
                "DATE": ["2026-03-01", "2026-03-01"],
                "FIGI": ["aapl_figi", "msft_figi"],
                "WEIGHT": [0.5, 0.5],
            }
        ),
    )
    monkeypatch.setattr(
        "data_loading.prices_data_source.get_prices",
        lambda **_: pd.DataFrame(
            {
                "DATE": [
                    "2026-03-01",
                    "2026-03-01",
                    "2026-03-02",
                    "2026-03-02",
                ],
                "FIGI": ["AAPL_FIGI", "MSFT_FIGI", "AAPL_FIGI", "MSFT_FIGI"],
                "ADJ_CLOSE": [100.0, 200.0, 101.0, 202.0],
            }
        ),
    )
    monkeypatch.setattr(
        "data_loading.company_info_data_source.get_company_info",
        lambda **_: pd.DataFrame(
            {
                "DATE": ["2026-03-01", "2026-03-01"],
                "FIGI": ["AAPL_FIGI", "MSFT_FIGI"],
                "TICKER": ["AAPL", "MSFT"],
                "SECTOR": ["Technology", "Technology"],
            }
        ),
    )

    runner = Runner(portfolio="S&P 500")
    runner.load_data()

    assert runner.figi_to_ticker == {
        "AAPL_FIGI": "AAPL",
        "MSFT_FIGI": "MSFT",
    }
    assert runner.ticker_to_figi == {
        "AAPL": "AAPL_FIGI",
        "MSFT": "MSFT_FIGI",
    }


def test_runner_builds_index_figi_ticker_maps_from_index_prices_long():
    runner = Runner(portfolio="S&P 500")
    runner.security_data_source = SimpleNamespace(formatted_data={})
    runner.index_data_source = SimpleNamespace(
        formatted_data={
            "index_prices_long": pd.DataFrame(
                {
                    "FIGI": ["FIGI_SPX", "FIGI_NDX"],
                    "TICKER": ["SPX", "NDX"],
                }
            )
        }
    )

    runner._build_symbol_mappings()

    assert runner.figi_to_ticker == {
        "FIGI_SPX": "SPX",
        "FIGI_NDX": "NDX",
    }
    assert runner.ticker_to_figi == {
        "SPX": "FIGI_SPX",
        "NDX": "FIGI_NDX",
    }


def test_runner_plot_prices_supports_tickers(monkeypatch):
    called = {"show": 0}

    def fake_show(self):  # noqa: ANN001
        called["show"] += 1

    monkeypatch.setattr("visualization.figure.Figure.show", fake_show)

    runner = Runner(portfolio="S&P 500")
    runner.prices_data_source = SimpleNamespace(
        formatted_data={
            "prices_wide": pd.DataFrame(
                {
                    "AAPL_FIGI": [100.0, 101.0],
                    "MSFT_FIGI": [200.0, 202.0],
                },
                index=pd.to_datetime(["2026-03-01", "2026-03-02"]),
            ),
            "prices_wide_full_history": pd.DataFrame(
                {
                    "AAPL_FIGI": [99.0, 100.0, 101.0],
                    "MSFT_FIGI": [198.0, 200.0, 202.0],
                },
                index=pd.to_datetime(["2026-02-28", "2026-03-01", "2026-03-02"]),
            ),
        }
    )
    runner.figi_to_ticker = {"AAPL_FIGI": "AAPL", "MSFT_FIGI": "MSFT"}
    runner.ticker_to_figi = {"AAPL": "AAPL_FIGI", "MSFT": "MSFT_FIGI"}

    fig = runner.plot_prices(tickers=["AAPL"], title="Prices")
    built = fig.build().fig

    assert built is not None
    assert len(built.data) == 1
    assert built.data[0].name == "AAPL"
    assert called["show"] == 1


def test_runner_plot_prices_supports_figis(monkeypatch):
    called = {"show": 0}

    def fake_show(self):  # noqa: ANN001
        called["show"] += 1

    monkeypatch.setattr("visualization.figure.Figure.show", fake_show)

    runner = Runner(portfolio="S&P 500")
    runner.prices_data_source = SimpleNamespace(
        formatted_data={
            "prices_wide": pd.DataFrame(
                {"AAPL_FIGI": [100.0, 101.0]},
                index=pd.to_datetime(["2026-03-01", "2026-03-02"]),
            )
        }
    )
    runner.figi_to_ticker = {"AAPL_FIGI": "AAPL"}
    runner.ticker_to_figi = {"AAPL": "AAPL_FIGI"}

    fig = runner.plot_prices(figis=["AAPL_FIGI"], title="Prices")
    built = fig.build().fig

    assert built is not None
    assert len(built.data) == 1
    assert built.data[0].name == "AAPL"
    assert called["show"] == 1
