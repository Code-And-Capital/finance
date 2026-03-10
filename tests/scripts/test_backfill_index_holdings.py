from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import types as satypes

import scripts.backfill_index_holdings as backfill_script


class _DummyRunner:
    last_init: dict | None = None
    payload: dict | None = None

    def __init__(
        self,
        *,
        portfolio,
        start_date,
        end_date,
        configs_path=None,
    ) -> None:
        self.__class__.last_init = {
            "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "configs_path": configs_path,
        }

    def run(self):
        if self.__class__.payload is None:
            raise RuntimeError("payload must be set for _DummyRunner")
        payload = self.__class__.payload
        self.holdings_data_source = SimpleNamespace(
            formatted_data=payload["holdings_formatted_data"]
        )
        self.prices_data_source = SimpleNamespace(
            formatted_data=payload["prices_formatted_data"]
        )
        return {
            "holdings": self.holdings_data_source,
            "prices": self.prices_data_source,
        }


class _DummyStrategy:
    last_instance: "_DummyStrategy | None" = None

    def __init__(self, *, name, algos) -> None:
        self.name = name
        self.algos = algos
        self.__class__.last_instance = self


class _DummyBacktest:
    last_instance: "_DummyBacktest | None" = None

    def __init__(self, *, strategy, data, integer_positions=False) -> None:
        self.strategy = strategy
        self.data = data
        self.integer_positions = integer_positions
        self.__class__.last_instance = self


def _build_runner_payload() -> dict:
    start = pd.Timestamp("2026-03-05")
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-03-04"),
            pd.Timestamp("2026-03-05"),
            pd.Timestamp("2026-03-06"),
            pd.Timestamp("2026-03-07"),
            pd.Timestamp("2026-03-08"),
        ]
    )
    figis = ["F1", "F2"]

    weights_wide = pd.DataFrame(
        {"F1": [0.6, 0.6, 0.5, 0.5], "F2": [0.4, 0.4, 0.5, 0.5]},
        index=pd.DatetimeIndex(
            [
                start,
                start + pd.Timedelta(days=1),
                start + pd.Timedelta(days=2),
                start + pd.Timedelta(days=3),
            ]
        ),
    )
    in_portfolio_wide = weights_wide > 0

    prices_wide_full_history = pd.DataFrame(
        {
            "F1": [100.0, 101.0, 102.0, 103.0, 104.0],
            "F2": [50.0, 51.0, 52.0, 53.0, 54.0],
        },
        index=dates,
    )

    holdings_long = pd.DataFrame(
        [
            {
                "DATE": "2026-03-04",
                "FIGI": "F1",
                "TICKER": "OLD1",
                "QUANTITY": 99.0,
                "WEIGHT": 0.5,
                "MARKET_VALUE": 0.0,
                "PRICE": 0.0,
            },
            {
                "DATE": "2026-03-05",
                "FIGI": "F1",
                "TICKER": "AAA",
                "QUANTITY": 10.0,
                "WEIGHT": 0.6,
                "MARKET_VALUE": 0.0,
                "PRICE": 0.0,
            },
            {
                "DATE": "2026-03-05",
                "FIGI": "F2",
                "TICKER": "BBB",
                "QUANTITY": 20.0,
                "WEIGHT": 0.4,
                "MARKET_VALUE": 0.0,
                "PRICE": 0.0,
            },
        ]
    )

    last_valid_date = pd.Series(
        {"F1": pd.Timestamp("2026-03-07"), "F2": pd.Timestamp("2026-03-08")}
    )

    return {
        "holdings_formatted_data": {
            "weights_wide": weights_wide,
            "in_portfolio_wide": in_portfolio_wide,
            "holdings_long": holdings_long,
        },
        "prices_formatted_data": {
            "prices_wide_full_history": prices_wide_full_history,
            "last_valid_date": last_valid_date,
        },
    }


def test_build_index_holdings_backfill_runner_flow_and_start_date_anchor(monkeypatch):
    _DummyRunner.payload = _build_runner_payload()
    monkeypatch.setattr(backfill_script, "Runner", _DummyRunner)
    monkeypatch.setattr(backfill_script, "Strategy", _DummyStrategy)
    monkeypatch.setattr(backfill_script, "Backtest", _DummyBacktest)

    security_weights = pd.DataFrame(
        {
            "F1": [0.6, 0.5, 0.5, 0.4],
            "F2": [0.4, 0.5, 0.5, 0.0],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2026-03-05"),
                pd.Timestamp("2026-03-06"),
                pd.Timestamp("2026-03-07"),
                pd.Timestamp("2026-03-08"),
            ]
        ),
    )

    def _dummy_run(_backtest):
        return SimpleNamespace(
            backtests={
                "buy_and_hold": SimpleNamespace(security_weights=security_weights)
            }
        )

    monkeypatch.setattr(backfill_script.bt, "run", _dummy_run)

    out = backfill_script.build_index_holdings_backfill(
        index_name="S&P 500",
        start_date="2026-03-05",
        end_date="2026-03-08",
        write_to_azure=False,
    )

    assert _DummyRunner.last_init == {
        "portfolio": ["S&P 500"],
        "start_date": "2026-03-05",
        "end_date": "2026-03-08",
        "configs_path": None,
    }
    assert _DummyStrategy.last_instance is not None
    algo_names = [
        type(algo_obj).__name__ for algo_obj in _DummyStrategy.last_instance.algos
    ]
    assert algo_names == [
        "ClosePositionsAfterDates",
        "RunAfterDate",
        "RunOnce",
        "SelectWhere",
        "SelectAll",
        "SelectActive",
        "WeightFixedSchedule",
        "Rebalance",
    ]

    # start_date_output should be 2026-03-06; one zero-weight row is filtered out.
    assert out["DATE"].min() == pd.Timestamp("2026-03-06")
    assert set(out["FIGI"]) == {"F1", "F2"}
    assert "OLD1" not in set(out["TICKER"])
    assert out["WEIGHT"].gt(0).all()
    assert (out["MARKET_VALUE"] == out["QUANTITY"] * out["PRICE"]).all()


def test_build_index_holdings_backfill_rejects_invalid_date_order():
    with pytest.raises(ValueError, match="end_date must be on or after start_date"):
        backfill_script.build_index_holdings_backfill(
            index_name="S&P 500",
            start_date="2026-03-08",
            end_date="2026-03-05",
        )


def test_build_index_holdings_backfill_rejects_insufficient_price_history(monkeypatch):
    payload = _build_runner_payload()
    payload["prices_formatted_data"]["prices_wide_full_history"] = pd.DataFrame(
        {"F1": [101.0], "F2": [51.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-05")]),
    )
    _DummyRunner.payload = payload
    monkeypatch.setattr(backfill_script, "Runner", _DummyRunner)

    with pytest.raises(ValueError, match="Insufficient price history"):
        backfill_script.build_index_holdings_backfill(
            index_name="S&P 500",
            start_date="2026-03-05",
            end_date="2026-03-08",
        )


def test_build_index_holdings_backfill_writes_to_azure(monkeypatch):
    _DummyRunner.payload = _build_runner_payload()
    monkeypatch.setattr(backfill_script, "Runner", _DummyRunner)
    monkeypatch.setattr(backfill_script, "Strategy", _DummyStrategy)
    monkeypatch.setattr(backfill_script, "Backtest", _DummyBacktest)

    security_weights = pd.DataFrame(
        {"F1": [0.5], "F2": [0.5]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-06")]),
    )
    monkeypatch.setattr(
        backfill_script.bt,
        "run",
        lambda _backtest: SimpleNamespace(
            backtests={
                "buy_and_hold": SimpleNamespace(security_weights=security_weights)
            }
        ),
    )

    calls: dict = {}

    def _get_engine(*, configs_path=None):
        calls["configs_path"] = configs_path
        return "ENGINE"

    def _write_sql_table(**kwargs):
        calls["write"] = kwargs

    monkeypatch.setattr(
        backfill_script.default_azure_data_source, "get_engine", _get_engine
    )
    monkeypatch.setattr(
        backfill_script.default_azure_data_source,
        "write_sql_table",
        _write_sql_table,
    )

    out = backfill_script.build_index_holdings_backfill(
        index_name="S&P 500",
        start_date="2026-03-05",
        end_date="2026-03-08",
        write_to_azure=True,
        configs_path="cfg/path.toml",
    )

    assert len(out) > 0
    assert calls["configs_path"] == "cfg/path.toml"
    assert calls["write"]["table_name"] == "holdings"
    assert calls["write"]["engine"] == "ENGINE"
    assert calls["write"]["overwrite"] is False
    assert isinstance(calls["write"]["dtype_overrides"]["DATE"], satypes.Date)
