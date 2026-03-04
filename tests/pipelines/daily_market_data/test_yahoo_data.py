from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from pipelines.daily_market_data.yahoo_data import YahooData


def test_resolve_client_uses_injected_client():
    injected = MagicMock()
    pipeline = YahooData(tickers=["AAPL"], client=injected)

    resolved = pipeline._resolve_client()

    assert resolved is injected


def test_resolve_client_caches_created_client():
    pipeline = YahooData(tickers=["AAPL"])
    created = MagicMock()
    pipeline._create_client = MagicMock(return_value=created)

    first = pipeline._resolve_client()
    second = pipeline._resolve_client()

    assert first is created
    assert second is created
    pipeline._create_client.assert_called_once()


def test_pull_generic_calls_client_method_and_coerces_existing_dates():
    client = MagicMock()
    client.get_prices.return_value = pd.DataFrame(
        {"DATE": ["2024-01-02"], "VALUE": [100.0]}
    )
    pipeline = YahooData(tickers=["AAPL"], client=client)

    out = pipeline._pull_generic(
        client_method="get_prices",
        method_kwargs={"start_date": "2024-01-01"},
        date_columns=["DATE", "MISSING_DATE"],
    )

    client.get_prices.assert_called_once_with(start_date="2024-01-01")
    assert pd.api.types.is_datetime64_any_dtype(out["DATE"])
    assert "VALUE" in out.columns


def test_create_client_for_tickers_tracks_clients():
    pipeline = YahooData(tickers=["AAPL"])

    first = pipeline._create_client_for_tickers(["AAPL"])
    second = pipeline._create_client_for_tickers(["MSFT"])

    assert len(pipeline.clients_used) == 2
    assert pipeline.clients_used[0] is first
    assert pipeline.clients_used[1] is second


def test_pull_generic_rate_limit_waits_and_retries():
    client = MagicMock()
    client.get_prices = MagicMock(
        side_effect=[
            RuntimeError("Too Many Requests. Rate limited. Try after a while."),
            pd.DataFrame({"DATE": ["2024-01-02"], "VALUE": [100.0]}),
        ]
    )
    pipeline = YahooData(tickers=["AAPL"], client=client)
    pipeline._sleep = MagicMock(return_value=None)

    with patch("pipelines.daily_market_data.yahoo_data.log") as mock_log:
        out = pipeline._pull_generic(
            client_method="get_prices",
            method_kwargs={},
            date_columns=["DATE"],
        )

    assert len(out) == 1
    assert client.get_prices.call_count == 2
    pipeline._sleep.assert_called_once_with(120)
    assert any(
        "Rate limit while calling Yahoo method 'get_prices'" in call.args[0]
        for call in mock_log.call_args_list
    )


def test_pull_generic_non_rate_limit_error_raises():
    import pytest

    client = MagicMock()
    client.get_prices = MagicMock(side_effect=RuntimeError("boom"))
    pipeline = YahooData(tickers=["AAPL"], client=client)
    pipeline._sleep = MagicMock(return_value=None)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline._pull_generic(client_method="get_prices")


def test_pull_with_missing_ticker_retries_retries_remaining_tickers():
    pipeline = YahooData(tickers=["AAPL", "MSFT"])
    first_client = MagicMock()
    second_client = MagicMock()
    pipeline._create_client_for_tickers = MagicMock(
        side_effect=[first_client, second_client]
    )
    pipeline._sleep = MagicMock(return_value=None)
    pipeline._pull_generic = MagicMock(
        side_effect=[
            pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]}),
            pd.DataFrame({"TICKER": ["MSFT"], "DATE": ["2024-01-02"]}),
        ]
    )

    out = pipeline._pull_with_missing_ticker_retries(
        client_method="get_info", wait_seconds=1, max_resets=2
    )

    assert len(out) == 2
    assert pipeline._create_client_for_tickers.call_args_list[0].args[0] == [
        "AAPL",
        "MSFT",
    ]
    assert pipeline._create_client_for_tickers.call_args_list[1].args[0] == ["MSFT"]
    pipeline._sleep.assert_not_called()


def test_pull_with_missing_ticker_retries_breaks_after_two_empty_attempts():
    pipeline = YahooData(tickers=["AAPL", "MSFT"])
    first_client = MagicMock()
    second_client = MagicMock()
    pipeline._create_client_for_tickers = MagicMock(
        side_effect=[first_client, second_client]
    )
    pipeline._sleep = MagicMock(return_value=None)
    pipeline._pull_generic = MagicMock(side_effect=[pd.DataFrame(), pd.DataFrame()])

    out = pipeline._pull_with_missing_ticker_retries(
        client_method="get_info", wait_seconds=1, max_resets=10
    )

    assert out.empty
    assert pipeline._pull_generic.call_count == 2
    pipeline._sleep.assert_not_called()


def test_pull_with_missing_ticker_retries_waits_when_many_tickers_remaining():
    tickers = [f"T{i:02d}" for i in range(1, 22)]
    pipeline = YahooData(tickers=tickers)
    first_client = MagicMock()
    second_client = MagicMock()
    pipeline._create_client_for_tickers = MagicMock(
        side_effect=[first_client, second_client]
    )
    pipeline._sleep = MagicMock(return_value=None)
    first_ticker = tickers[0]
    remaining = tickers[1:]
    pipeline._pull_generic = MagicMock(
        side_effect=[
            pd.DataFrame({"TICKER": [first_ticker], "DATE": ["2024-01-02"]}),
            pd.DataFrame(
                {"TICKER": remaining, "DATE": ["2024-01-02"] * len(remaining)}
            ),
        ]
    )

    out = pipeline._pull_with_missing_ticker_retries(
        client_method="get_info", wait_seconds=1, max_resets=2
    )

    assert len(out) == len(tickers)
    pipeline._sleep.assert_called_once_with(1)


def test_pull_with_missing_ticker_retries_keeps_partial_rows_on_rate_limit():
    pipeline = YahooData(tickers=["AAPL", "MSFT"])
    first_client = MagicMock()
    second_client = MagicMock()
    pipeline._create_client_for_tickers = MagicMock(
        side_effect=[first_client, second_client]
    )
    pipeline._sleep = MagicMock(return_value=None)

    rate_limit_exc = RuntimeError("Too Many Requests. Rate limited. Try after a while.")
    setattr(
        rate_limit_exc,
        "partial_df",
        pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"]}),
    )
    pipeline._pull_generic = MagicMock(
        side_effect=[
            rate_limit_exc,
            pd.DataFrame({"TICKER": ["MSFT"], "DATE": ["2024-01-02"]}),
        ]
    )

    out = pipeline._pull_with_missing_ticker_retries(
        client_method="get_info", wait_seconds=1, max_resets=2
    )

    assert len(out) == 2
    assert set(out["TICKER"]) == {"AAPL", "MSFT"}
    assert pipeline._create_client_for_tickers.call_args_list[0].args[0] == [
        "AAPL",
        "MSFT",
    ]
    assert pipeline._create_client_for_tickers.call_args_list[1].args[0] == ["MSFT"]
    pipeline._sleep.assert_not_called()


def test_filter_new_or_changed_rows_ignores_date_and_keeps_only_delta():
    pipeline = YahooData(tickers=["AAPL"])
    existing = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "NAME": ["Tim Cook"],
            "TITLE": ["CEO"],
            "DATE": ["2024-01-01"],
        }
    )
    incoming = pd.DataFrame(
        {
            "TICKER": ["AAPL", "AAPL", "MSFT"],
            "NAME": ["Tim Cook", "Tim Cook", "Satya Nadella"],
            "TITLE": ["CEO", "CEO", "CEO"],
            "DATE": ["2024-01-02", "2024-01-03", "2024-01-02"],
        }
    )

    out = pipeline._filter_new_or_changed_rows(
        incoming_df=incoming,
        existing_df=existing,
        exclude_columns={"DATE"},
    )

    assert len(out) == 1
    assert out.iloc[0]["TICKER"] == "MSFT"


def test_filter_new_or_changed_rows_normalizes_equivalent_types():
    pipeline = YahooData(tickers=["AAPL"])
    existing = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "IN_THE_MONEY": [1],
            "STRIKE": ["100"],
            "LASTTRADEDATE": ["2026-02-28"],
            "DATE": ["2026-03-01"],
        }
    )
    incoming = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "IN_THE_MONEY": [True],
            "STRIKE": [100.0],
            "LASTTRADEDATE": [pd.Timestamp("2026-02-28 00:00:00+00:00")],
            "DATE": ["2026-03-02"],
        }
    )

    out = pipeline._filter_new_or_changed_rows(
        incoming_df=incoming,
        existing_df=existing,
        exclude_columns={"DATE"},
    )

    assert out.empty


def test_filter_new_or_changed_rows_treats_date_and_timestamp_as_equal():
    pipeline = YahooData(tickers=["AAPL"])
    existing = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "EXPIRATION": [pd.Timestamp("2026-02-28 00:00:00")],
            "DATE": ["2026-03-01"],
        }
    )
    incoming = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "EXPIRATION": [pd.Timestamp("2026-02-28").date()],
            "DATE": ["2026-03-02"],
        }
    )

    out = pipeline._filter_new_or_changed_rows(
        incoming_df=incoming,
        existing_df=existing,
        exclude_columns={"DATE"},
    )

    assert out.empty


def test_filter_new_or_changed_rows_ignores_time_for_datetime_values():
    pipeline = YahooData(tickers=["AAPL"])
    existing = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "LASTTRADEDATE": [pd.Timestamp("2026-03-01 00:00:00")],
            "DATE": ["2026-03-01"],
        }
    )
    incoming = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "LASTTRADEDATE": [pd.Timestamp("2026-03-01 15:47:12")],
            "DATE": ["2026-03-02"],
        }
    )

    out = pipeline._filter_new_or_changed_rows(
        incoming_df=incoming,
        existing_df=existing,
        exclude_columns={"DATE"},
    )

    assert out.empty
