from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from connectors.yahoo_data_source import YahooDataClient


@pytest.fixture
def mock_ticker_objects() -> dict[str, MagicMock]:
    return {
        "AAPL": MagicMock(name="AAPL_OBJ"),
        "MSFT": MagicMock(name="MSFT_OBJ"),
    }


@pytest.fixture
def mock_yf(
    monkeypatch: pytest.MonkeyPatch, mock_ticker_objects: dict[str, MagicMock]
) -> dict[str, MagicMock]:
    batch = SimpleNamespace(tickers=mock_ticker_objects)
    monkeypatch.setattr("connectors.yahoo_data_source.yf.Tickers", lambda _: batch)
    return mock_ticker_objects


@pytest.fixture
def mock_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyPool:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def run(
            self,
            tasks,
            return_exceptions: bool = False,
            stop_on_exception: bool = False,
        ):
            results = []
            for task in tasks:
                try:
                    results.append(task())
                except Exception as exc:  # noqa: BLE001
                    if stop_on_exception:
                        raise
                    results.append(exc if return_exceptions else None)
            return results

    monkeypatch.setattr("connectors.yahoo_data_source.ThreadWorkerPool", DummyPool)


@pytest.fixture
def mock_log(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr("connectors.yahoo_data_source.log", mock)
    return mock


@pytest.fixture
def fixed_now(monkeypatch: pytest.MonkeyPatch) -> pd.Timestamp:
    ts = pd.Timestamp("2024-01-01 09:30:00")
    monkeypatch.setattr(
        "connectors.yahoo_data_source.pd.Timestamp.now", lambda *_, **__: ts
    )
    return ts


def build_client(
    mock_yf: dict[str, MagicMock],
    mock_pool: None,
    tickers: list[str] | None = None,
    **kwargs,
) -> YahooDataClient:
    return YahooDataClient(tickers or ["AAPL", "MSFT"], **kwargs)


def test_init_normalizes_deduplicates_and_sets_config(mock_yf, mock_pool):
    client = build_client(
        mock_yf, mock_pool, tickers=["aapl", " AAPL ", "msft"], max_workers=4, retries=2
    )

    assert client.tickers == ["AAPL", "MSFT"]
    assert client.max_workers == 4
    assert client.retries == 2


@pytest.mark.parametrize(
    "tickers,max_workers,retries,expected",
    [
        ("AAPL", 8, 3, "tickers must be a sequence"),
        ([], 8, 3, "tickers cannot be empty"),
        (["AAPL"], 0, 3, "max_workers must be >= 1"),
        (["AAPL"], 8, 0, "retries must be >= 1"),
    ],
)
def test_init_validation_errors(
    mock_yf, mock_pool, tickers, max_workers, retries, expected
):
    with pytest.raises(ValueError, match=expected):
        YahooDataClient(tickers=tickers, max_workers=max_workers, retries=retries)


def test_resolve_start_date_scalar_and_mapping(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    assert client._resolve_start_date("2020-01-01", "AAPL") == "2020-01-01"
    assert client._resolve_start_date({"AAPL": "2021-01-01"}, "AAPL") == "2021-01-01"
    assert client._resolve_start_date({"AAPL": "2021-01-01"}, "MSFT") is None


def test_retry_fetch_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, mock_yf, mock_pool
):
    client = build_client(mock_yf, mock_pool, retries=3)

    sleep = MagicMock()
    monkeypatch.setattr("connectors.yahoo_data_source.time.sleep", sleep)

    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary")
        return pd.DataFrame({"ok": [1]})

    out = client._retry_fetch(flaky, "AAPL")
    assert isinstance(out, pd.DataFrame)
    assert calls["count"] == 3
    assert sleep.call_count == 2


def test_retry_fetch_raises_without_logging(
    monkeypatch: pytest.MonkeyPatch, mock_yf, mock_pool, mock_log
):
    client = build_client(mock_yf, mock_pool, retries=2)

    sleep = MagicMock()
    monkeypatch.setattr("connectors.yahoo_data_source.time.sleep", sleep)

    def always_fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        client._retry_fetch(always_fail, "AAPL")

    assert sleep.call_count == 1
    mock_log.assert_not_called()


def test_add_metadata_inserts_leading_columns(mock_yf, mock_pool, fixed_now):
    _ = fixed_now
    df = pd.DataFrame({"x": [1]})

    out = YahooDataClient._add_metadata(df, "AAPL")

    assert list(out.columns[:2]) == ["DATE", "TICKER"]
    assert out.loc[0, "DATE"] == pd.Timestamp("2024-01-01 09:30:00")
    assert out.loc[0, "TICKER"] == "AAPL"


def test_normalize_deduplicates_colliding_columns(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    df = pd.DataFrame([[1, 2]], columns=["date", "DATE"])

    out = client._normalize(df)

    assert list(out.columns) == ["DATE", "DATE_2"]


def test_fetch_info_success_removes_company_officers(mock_yf, mock_pool, fixed_now):
    _ = fixed_now
    client = build_client(mock_yf, mock_pool)
    obj = mock_yf["AAPL"]
    obj.info = {"industry": "Tech", "companyOfficers": [{"name": "CEO"}]}

    out = client._fetch_info("AAPL", obj)

    assert isinstance(out, pd.DataFrame)
    assert "COMPANYOFFICERS" not in out.columns
    assert out.loc[0, "TICKER"] == "AAPL"


def test_fetch_officers_success_and_empty(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    obj = mock_yf["AAPL"]

    obj.info = {"companyOfficers": [{"name": "CEO"}]}
    out = client._fetch_officers("AAPL", obj)
    assert out is not None
    assert out.loc[0, "NAME"] == "CEO"

    obj.info = {"companyOfficers": []}
    assert client._fetch_officers("AAPL", obj) is None


def test_fetch_officers_reflects_current_info_payload(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    obj = mock_yf["AAPL"]

    obj.info = {"industry": "Tech", "companyOfficers": [{"name": "CEO"}]}
    officers_df = client._fetch_officers("AAPL", obj)
    assert officers_df is not None
    assert officers_df.loc[0, "NAME"] == "CEO"

    obj.info = {"industry": "Tech", "companyOfficers": []}
    assert client._fetch_officers("AAPL", obj) is None


def test_fetch_prices_uses_per_ticker_start_date_and_normalizes_duplicate_date(
    mock_yf, mock_pool, fixed_now
):
    client = build_client(mock_yf, mock_pool)
    obj = mock_yf["AAPL"]

    obj.history.return_value = pd.DataFrame(
        {"Date": [pd.Timestamp("2024-01-02")], "Open": [10.0], "Close": [11.0]}
    )

    out = client._fetch_prices("AAPL", obj, {"AAPL": "2020-01-01"})

    obj.history.assert_called_once_with(start="2020-01-01", auto_adjust=False)
    assert out is not None
    assert out.columns.is_unique
    assert "DATE" in out.columns
    assert "DATE_2" not in out.columns
    assert "TICKER" in out.columns
    assert out.loc[0, "TICKER"] == "AAPL"


def test_fetch_table_attr_dataframe_and_list(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    obj = mock_yf["AAPL"]

    obj.recommendations = pd.DataFrame(
        {"To Grade": ["Buy"]}, index=[pd.Timestamp("2024-01-01")]
    )
    out_df = client._fetch_table_attr("AAPL", obj, "recommendations")
    assert out_df is not None
    assert "TO_GRADE" in out_df.columns

    obj.some_list_attr = [{"k": 1}]
    out_list = client._fetch_table_attr(
        "AAPL", obj, "some_list_attr", reset_index=False
    )
    assert out_list is not None
    assert "K" in out_list.columns


def test_fetch_table_attr_missing_or_empty_returns_none(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    obj = mock_yf["AAPL"]

    assert client._fetch_table_attr("AAPL", obj, "missing_attr") is None

    obj.empty_df = pd.DataFrame()
    assert client._fetch_table_attr("AAPL", obj, "empty_df") is None


def test_fetch_financials_success_and_invalid_type(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    obj = mock_yf["AAPL"]

    obj.financials = pd.DataFrame(
        {pd.Timestamp("2022-12-31"): [100]},
        index=["Revenue"],
    )

    out = client._fetch_financials("AAPL", obj, "financial", True)
    assert out is not None
    assert "REPORT_DATE" in out.columns
    assert "REVENUE" in out.columns

    with pytest.raises(ValueError, match="statement_type must be one of"):
        client._fetch_financials("AAPL", obj, "bad_type", True)


def test_fetch_options_builds_calls_and_puts(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    obj = mock_yf["AAPL"]

    obj.options = ["2024-12-20"]
    obj.option_chain.return_value = SimpleNamespace(
        calls=pd.DataFrame({"strike": [100]}),
        puts=pd.DataFrame({"strike": [100]}),
    )

    out = client._fetch_options("AAPL", obj)

    assert out is not None
    assert set(out["OPTION_TYPE"]) == {"Call", "Put"}
    assert "EXPIRATION" in out.columns


def test_fetch_insider_transactions_classification(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    obj = mock_yf["AAPL"]

    obj.insider_transactions = pd.DataFrame(
        {
            "Text": [
                "Open market sale",
                "gift to family",
                "grant award",
                "conversion event",
                "purchase on market",
                None,
            ]
        }
    )

    out = client._fetch_insider_transactions("AAPL", obj)

    assert out is not None
    assert list(out["TRANSACTION"]) == [
        "SELL",
        "GIFT",
        "GRANT",
        "CONVERSION",
        "BUY",
        "BUY",
    ]


def test_fetch_analyst_estimate_success_and_none(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    obj = mock_yf["AAPL"]

    obj.earnings_estimate = pd.DataFrame({"avg": [2.5]}, index=["0q"])
    out = client._fetch_analyst_estimate("AAPL", obj, "eps")
    assert out is not None
    assert out.loc[0, "ESTIMATE_TYPE"] == "EPS"

    obj.earnings_estimate = pd.DataFrame()
    assert client._fetch_analyst_estimate("AAPL", obj, "eps") is None


def test_run_parallel_concatenates_only_non_empty_frames(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)

    def fetcher(ticker: str, _obj: MagicMock):
        if ticker == "AAPL":
            return pd.DataFrame({"X": [1]})
        return None

    out = client._run_parallel(fetcher, "Loading Test")
    assert len(out) == 1


def test_run_parallel_raises_on_rate_limit_exceptions(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)

    def fetcher(_ticker: str, _obj: MagicMock):
        raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")

    with pytest.raises(RuntimeError, match="Too Many Requests"):
        client._run_parallel(fetcher, "Loading Test")


def test_concat_results_keeps_all_na_columns(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    first = pd.DataFrame({"A": [1], "ALL_NA": [pd.NA]})
    second = pd.DataFrame({"A": [2], "ALL_NA": [pd.NA]})

    out = client._concat_results([first, second])

    assert "ALL_NA" in out.columns
    assert len(out) == 2


def test_iter_ticker_objects_creates_missing_subset_ticker(
    mock_yf, mock_pool, monkeypatch: pytest.MonkeyPatch
):
    client = build_client(mock_yf, mock_pool)
    created_obj = MagicMock(name="GOOG_OBJ")
    ticker_ctor = MagicMock(return_value=created_obj)
    monkeypatch.setattr("connectors.yahoo_data_source.yf.Ticker", ticker_ctor)

    pairs = client._iter_ticker_objects(["GOOG"])

    ticker_ctor.assert_called_once_with("GOOG")
    assert pairs[0][0] == "GOOG"
    assert pairs[0][1] is created_obj
    assert client.yf_obj.tickers["GOOG"] is created_obj


@pytest.mark.parametrize(
    "method_name,private_name,args,expected_calls",
    [
        ("get_company_info", "_fetch_info", (), 2),
        ("get_officer_info", "_fetch_officers", (), 2),
        ("get_options", "_fetch_options", (), 2),
        ("get_analyst_price_targets", "_fetch_analyst_price_target", (), 2),
        ("get_insider_transactions", "_fetch_insider_transactions", (), 2),
    ],
)
def test_public_methods_using_direct_fetchers(
    mock_yf, mock_pool, method_name, private_name, args, expected_calls
):
    client = build_client(mock_yf, mock_pool)
    setattr(client, private_name, MagicMock(return_value=pd.DataFrame({"X": [1]})))

    out = getattr(client, method_name)(*args)

    assert len(out) == expected_calls
    assert getattr(client, private_name).call_count == expected_calls


def test_get_prices_routes_start_date(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    client._fetch_prices = MagicMock(return_value=pd.DataFrame({"X": [1]}))

    out = client.get_prices(start_date={"AAPL": "2015-01-01"})

    assert len(out) == 2
    assert client._fetch_prices.call_count == 2
    for call in client._fetch_prices.call_args_list:
        assert call.args[2] == {"AAPL": "2015-01-01"}


def test_get_financials_routes_args(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    client._fetch_financials = MagicMock(return_value=pd.DataFrame({"X": [1]}))

    out = client.get_financials(statement_type="cashflow", annual=False)

    assert len(out) == 2
    for call in client._fetch_financials.call_args_list:
        assert call.args[2] == "cashflow"
        assert call.args[3] is False


def test_get_analyst_estimates_routes_and_validates(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)
    client._fetch_analyst_estimate = MagicMock(return_value=pd.DataFrame({"X": [1]}))

    out = client.get_analyst_estimates("eps")

    assert len(out) == 2
    for call in client._fetch_analyst_estimate.call_args_list:
        assert call.args[2] == "eps"


def test_get_analyst_estimates_invalid_type_raises(mock_yf, mock_pool):
    client = build_client(mock_yf, mock_pool)

    with pytest.raises(ValueError, match="estimate_type must be one of"):
        client.get_analyst_estimates("bad")


@pytest.mark.parametrize(
    "method_name,attr_name",
    [
        ("get_actions", "actions"),
        ("get_recommendations", "recommendations"),
        ("get_upgrades_downgrades", "upgrades_downgrades"),
        ("get_eps_revisions", "eps_revisions"),
        ("get_earnings_surprises", "earnings_dates"),
        ("get_institutional_holders", "institutional_holders"),
        ("get_major_holders", "major_holders"),
    ],
)
def test_public_methods_using_table_attr(mock_yf, mock_pool, method_name, attr_name):
    client = build_client(mock_yf, mock_pool)
    client._fetch_table_attr = MagicMock(return_value=pd.DataFrame({"X": [1]}))

    out = getattr(client, method_name)()

    assert len(out) == 2
    assert client._fetch_table_attr.call_count == 2
    for call in client._fetch_table_attr.call_args_list:
        assert call.args[2] == attr_name


def test_is_rate_limit_error_treats_earnings_date_issue_as_retriable(
    mock_yf,
    mock_pool,
):
    client = build_client(mock_yf, mock_pool)
    assert client._is_rate_limit_error(KeyError(["Earnings Date"])) is True
