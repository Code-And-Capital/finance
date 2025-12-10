import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import date

from data_loader.yahoo_finance import YahooDataClient


# ---------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------


@pytest.fixture
def mock_normalize(monkeypatch):
    """Mock dataframe_utils.normalize_columns to return df unchanged."""
    monkeypatch.setattr("utils.dataframe_utils.normalize_columns", lambda df: df)


@pytest.fixture
def mock_pool(monkeypatch):
    """Mock ThreadWorkerPool.run to synchronously execute tasks."""

    class DummyPool:
        def run(self, tasks):
            return [fn() for fn in tasks]

    monkeypatch.setattr(
        "utils.threading.ThreadWorkerPool", lambda max_workers: DummyPool()
    )
    return DummyPool()


@pytest.fixture
def mock_today(monkeypatch):
    """Fix today's date for deterministic tests."""
    monkeypatch.setattr(
        "data_loader.yahoo_finance.date", MagicMock(today=lambda: date(2020, 1, 1))
    )


@pytest.fixture
def mock_logging(monkeypatch):
    monkeypatch.setattr("utils.logging.log", lambda *args, **kwargs: None)


@pytest.fixture
def mock_yf(monkeypatch):
    """Mock yfinance Tickers and per-ticker objects."""
    mock_obj_A = MagicMock()
    mock_obj_B = MagicMock()

    tickers_dict = {"A": mock_obj_A, "B": mock_obj_B}

    mock_tickers = MagicMock()
    mock_tickers.tickers = tickers_dict

    monkeypatch.setattr("yfinance.Tickers", lambda s: mock_tickers)

    return tickers_dict


# ---------------------------------------------------------
# BASIC CONSTRUCTOR TEST
# ---------------------------------------------------------


def test_init_valid_input(mock_pool, mock_yf):
    client = YahooDataClient(["A", "B"])
    assert client.tickers == ["A", "B"]
    assert client.max_workers == 8
    assert client.retries == 3


def test_init_invalid_input():
    with pytest.raises(ValueError):
        YahooDataClient("AAPL")  # not list/ndarray


# ---------------------------------------------------------
# _retry_fetch TESTS
# ---------------------------------------------------------


def test_retry_fetch_success(mock_logging):
    client = YahooDataClient(["A"])

    f = MagicMock(return_value=123)
    out = client._retry_fetch(f, "A")
    assert out == 123


def test_retry_fetch_retries_then_success(monkeypatch, mock_logging):
    client = YahooDataClient(["A"], retries=3)

    calls = {"n": 0}

    def failing_then_ok():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("temp")
        return "OK"

    out = client._retry_fetch(failing_then_ok, "A")
    assert out == "OK"
    assert calls["n"] == 3


def test_retry_fetch_failure_after_all_retries(mock_logging):
    client = YahooDataClient(["A"], retries=2)

    def always_fails():
        raise RuntimeError("bad")

    with pytest.raises(RuntimeError):
        client._retry_fetch(always_fails, "A")


# ---------------------------------------------------------
# add_metadata TEST
# ---------------------------------------------------------


def test_add_metadata_order(mock_today):
    df = pd.DataFrame({"x": [1], "y": [2]})
    out = YahooDataClient.add_metadata(df, "AAPL")

    assert list(out.columns[:2]) == ["DATE", "TICKER"]
    assert out.iloc[0]["DATE"] == date(2020, 1, 1)
    assert out.iloc[0]["TICKER"] == "AAPL"


# ---------------------------------------------------------
# _fetch_info TEST
# ---------------------------------------------------------


def test_fetch_info_success(mock_yf, mock_pool, mock_normalize, mock_today):
    client = YahooDataClient(["A"])
    obj = mock_yf["A"]

    obj.info = {"industry": "Tech", "employees": 10, "companyOfficers": [{"x": 1}]}

    df = client._fetch_info("A", obj)

    assert isinstance(df, pd.DataFrame)
    assert "industry" in df.columns
    assert "companyOfficers" not in df.columns  # removed
    assert df["TICKER"].iloc[0] == "A"
    assert df["DATE"].iloc[0] == date(2020, 1, 1)


def test_fetch_info_none(mock_yf, mock_pool):
    client = YahooDataClient(["A"])
    obj = mock_yf["A"]

    obj.info = None  # triggers error → retry → error

    with pytest.raises(NotImplementedError):
        client._fetch_info("A", obj)


# ---------------------------------------------------------
# _fetch_officers TEST
# ---------------------------------------------------------


def test_fetch_officers_has_data(mock_yf, mock_pool, mock_normalize, mock_today):
    client = YahooDataClient(["A"])
    obj = mock_yf["A"]

    obj.info = {"companyOfficers": [{"name": "CEO"}]}

    df = client._fetch_officers("A", obj)
    assert isinstance(df, pd.DataFrame)
    assert df["name"].iloc[0] == "CEO"


def test_fetch_officers_none(mock_yf, mock_pool):
    client = YahooDataClient(["A"])
    obj = mock_yf["A"]

    obj.info = {"companyOfficers": []}

    out = client._fetch_officers("A", obj)
    assert out is None


# ---------------------------------------------------------
# _fetch_prices TEST
# ---------------------------------------------------------


def test_fetch_prices_success(mock_yf, mock_pool, mock_normalize):
    client = YahooDataClient(["A"])
    obj = mock_yf["A"]

    df_hist = pd.DataFrame(
        {"DATE": pd.to_datetime(["2020-01-01"]), "OPEN": [1], "CLOSE": [2]}
    )

    obj.history.return_value = df_hist

    out = client._fetch_prices("A", obj, "2020-01-01")
    assert isinstance(out, pd.DataFrame)
    assert out["TICKER"].iloc[0] == "A"


def test_fetch_prices_empty(mock_yf, mock_pool):
    client = YahooDataClient(["A"])
    obj = mock_yf["A"]
    obj.history.return_value = pd.DataFrame()

    out = client._fetch_prices("A", obj, "2020-01-01")
    assert out is None


# ---------------------------------------------------------
# _fetch_actions TEST
# ---------------------------------------------------------


def test_fetch_actions_success(mock_yf, mock_pool, mock_normalize):
    client = YahooDataClient(["A"])
    obj = mock_yf["A"]

    actions = pd.DataFrame({"Dividends": [0.5]}, index=[pd.Timestamp("2020-01-01")])
    obj.actions = actions

    df = client._fetch_actions("A", obj)

    assert isinstance(df, pd.DataFrame)
    assert "Dividends" in df.columns


def test_fetch_actions_empty(mock_yf, mock_pool):
    client = YahooDataClient(["A"])
    obj = mock_yf["A"]
    obj.actions = pd.DataFrame()

    out = client._fetch_actions("A", obj)
    assert out is None


# ---------------------------------------------------------
# _fetch_financials TEST
# ---------------------------------------------------------


def test_fetch_financials_success(mock_yf, mock_pool, mock_normalize, mock_today):
    client = YahooDataClient(["A"])
    obj = mock_yf["A"]

    obj.financials = pd.DataFrame({"2020": [100], "2021": [200]})

    df = client._fetch_financials("A", obj, "financial", True)
    assert isinstance(df, pd.DataFrame)
    assert "REPORT_DATE" in df.columns
    assert df["TICKER"].iloc[0] == "A"


def test_fetch_financials_empty(mock_yf, mock_pool, mock_normalize):
    client = YahooDataClient(["A"])
    obj = mock_yf["A"]
    obj.financials = pd.DataFrame()

    out = client._fetch_financials("A", obj, "financial", True)
    assert out is None


# ---------------------------------------------------------
# PUBLIC API METHODS
# ---------------------------------------------------------


def test_get_company_info(mock_yf, mock_pool, mock_normalize, mock_today):
    client = YahooDataClient(["A", "B"])

    # Mock _fetch_info
    client._fetch_info = MagicMock(return_value=pd.DataFrame({"x": [1]}))

    df = client.get_company_info()
    assert len(df) == 2
    assert client._fetch_info.call_count == 2


def test_get_officer_info(mock_yf, mock_pool, mock_normalize):
    client = YahooDataClient(["A", "B"])
    client._fetch_officers = MagicMock(return_value=pd.DataFrame({"x": [1]}))

    df = client.get_officer_info()
    assert len(df) == 2
    assert client._fetch_officers.call_count == 2


def test_get_prices(mock_yf, mock_pool, mock_normalize):
    client = YahooDataClient(["A", "B"])
    client._fetch_prices = MagicMock(return_value=pd.DataFrame({"p": [1]}))

    df = client.get_prices()
    assert len(df) == 2
    assert client._fetch_prices.call_count == 2


def test_get_actions(mock_yf, mock_pool, mock_normalize):
    client = YahooDataClient(["A", "B"])
    client._fetch_actions = MagicMock(return_value=pd.DataFrame({"a": [1]}))

    df = client.get_actions()
    assert len(df) == 2
    assert client._fetch_actions.call_count == 2


def test_get_financials(mock_yf, mock_pool, mock_normalize):
    client = YahooDataClient(["A", "B"])
    client._fetch_financials = MagicMock(return_value=pd.DataFrame({"f": [1]}))

    df = client.get_financials("financial", True)
    assert len(df) == 2
    assert client._fetch_financials.call_count == 2
