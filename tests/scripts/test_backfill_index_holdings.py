from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import types as satypes

import scripts.backfill_index_holdings as backfill_script
from bt.core import Strategy


class _DummyRunner:
    last_init: dict | None = None
    payload: dict | None = None
    load_called = False
    last_strategy: Strategy | None = None
    last_run_kwargs: dict | None = None

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

    def load_data(self):
        if self.__class__.payload is None:
            raise RuntimeError("payload must be set for _DummyRunner")
        self.__class__.load_called = True
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

    def run_backtest(self, *, strategies):
        self.__class__.last_strategy = strategies
        self.__class__.last_run_kwargs = {"strategies": strategies}
        payload = self.__class__.payload
        return SimpleNamespace(
            backtests={
                "buy_and_hold": SimpleNamespace(
                    security_weights=payload["security_weights"]
                )
            }
        )


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
        "security_weights": security_weights,
    }


def test_build_index_holdings_backfill_uses_runner_load_and_run_backtest(monkeypatch):
    _DummyRunner.payload = _build_runner_payload()
    _DummyRunner.load_called = False
    _DummyRunner.last_strategy = None
    monkeypatch.setattr(backfill_script, "Runner", _DummyRunner)

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
    assert _DummyRunner.load_called is True
    assert isinstance(_DummyRunner.last_strategy, Strategy)
    assert _DummyRunner.last_strategy.name == "buy_and_hold"
    algo_names = [
        type(algo).__name__ for algo in _DummyRunner.last_strategy.stack.algos
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

    write_calls: list[dict] = []

    class _DummyAzureDataSource:
        def get_engine(self, configs_path=None):
            return "ENGINE"

        def write_sql_table(
            self,
            *,
            table_name,
            engine,
            df,
            overwrite,
            dtype_overrides,
        ):
            write_calls.append(
                {
                    "table_name": table_name,
                    "engine": engine,
                    "df": df.copy(),
                    "overwrite": overwrite,
                    "dtype_overrides": dtype_overrides,
                }
            )

    monkeypatch.setattr(
        backfill_script, "default_azure_data_source", _DummyAzureDataSource()
    )

    out = backfill_script.build_index_holdings_backfill(
        index_name="S&P 500",
        start_date="2026-03-05",
        end_date="2026-03-08",
        write_to_azure=True,
        configs_path="config.yml",
    )

    assert len(write_calls) == 1
    call = write_calls[0]
    assert call["table_name"] == "holdings"
    assert call["engine"] == "ENGINE"
    assert call["overwrite"] is False
    assert list(call["dtype_overrides"]) == ["DATE"]
    assert isinstance(call["dtype_overrides"]["DATE"], satypes.Date)
    pd.testing.assert_frame_equal(call["df"].reset_index(drop=True), out)
