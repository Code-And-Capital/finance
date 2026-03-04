from unittest.mock import MagicMock

import pandas as pd

from pipelines.daily_market_data.financial_data import FinancialData


def test_financial_data_run_pulls_all_datasets_in_order():
    pipeline = FinancialData(tickers=["AAPL", "MSFT"])
    pipeline.pause_seconds = 0
    pipeline._pull_with_missing_ticker_retries = MagicMock(
        side_effect=[
            pd.DataFrame({"X": [1]}),
            pd.DataFrame({"X": [2]}),
            pd.DataFrame({"X": [3]}),
            pd.DataFrame({"X": [4]}),
            pd.DataFrame({"X": [5]}),
            pd.DataFrame({"X": [6]}),
            pd.DataFrame({"X": [7]}),
            pd.DataFrame({"X": [8]}),
        ]
    )

    result = pipeline.run()

    expected_keys = {
        "financial_annual",
        "financial_quarterly",
        "balance_sheet_annual",
        "balance_sheet_quarterly",
        "income_statement_annual",
        "income_statement_quarterly",
        "cashflow_annual",
        "cashflow_quarterly",
    }
    assert set(result.keys()) == expected_keys
    assert pipeline._pull_with_missing_ticker_retries.call_count == 8

    first_kwargs = pipeline._pull_with_missing_ticker_retries.call_args_list[0].kwargs[
        "method_kwargs"
    ]
    second_kwargs = pipeline._pull_with_missing_ticker_retries.call_args_list[1].kwargs[
        "method_kwargs"
    ]
    assert first_kwargs == {"statement_type": "financial", "annual": True}
    assert second_kwargs == {"statement_type": "financial", "annual": False}


def test_financial_data_run_sleeps_between_calls():
    pipeline = FinancialData(tickers=["AAPL"])
    pipeline.pause_seconds = 2
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pd.DataFrame())
    pipeline._sleep = MagicMock(return_value=None)

    pipeline.run()

    assert pipeline._sleep.call_count == 7
    assert all(call.args[0] == 2 for call in pipeline._sleep.call_args_list)


def test_warns_for_missing_tickers_per_statement():
    pipeline = FinancialData(tickers=["AAPL", "MSFT"])
    pipeline.pause_seconds = 0
    pipeline._pull_with_missing_ticker_retries = MagicMock(
        side_effect=[
            pd.DataFrame({"TICKER": ["AAPL"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
        ]
    )

    from unittest.mock import patch

    with patch("pipelines.daily_market_data.financial_data.log") as mock_log:
        pipeline.run()

    assert any(
        "Missing tickers for financial (annual): 1 -> ['MSFT']" in call.args[0]
        for call in mock_log.call_args_list
    )


def test_warns_for_missing_tickers_when_ticker_column_absent():
    pipeline = FinancialData(tickers=["AAPL", "MSFT"])
    pipeline.pause_seconds = 0
    pipeline._pull_with_missing_ticker_retries = MagicMock(
        side_effect=[
            pd.DataFrame({"VALUE": [1]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
            pd.DataFrame({"TICKER": ["AAPL", "MSFT"]}),
        ]
    )

    from unittest.mock import patch

    with patch("pipelines.daily_market_data.financial_data.log") as mock_log:
        pipeline.run()

    assert any(
        "Missing tickers for financial (annual): 2 -> ['AAPL', 'MSFT']" in call.args[0]
        for call in mock_log.call_args_list
    )


def test_run_write_to_azure_writes_all_financial_tables():
    pipeline = FinancialData(tickers=["AAPL"])
    pipeline.pause_seconds = 0
    pipeline._pull_with_missing_ticker_retries = MagicMock(
        return_value=pd.DataFrame({"TICKER": ["AAPL"]})
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    result = pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    pipeline.azure_data_source.get_engine.assert_called_once_with(
        configs_path="config/configs.json"
    )
    assert len(result) == 8
    assert pipeline.azure_data_source.write_sql_table.call_count == 8
    assert pipeline.azure_data_source.read_sql_table.call_count == 8

    table_names = [
        call.kwargs["table_name"]
        for call in pipeline.azure_data_source.write_sql_table.call_args_list
    ]
    assert table_names == [
        "financial_annual",
        "financial_quarterly",
        "balancesheet_annual",
        "balancesheet_quarterly",
        "incomestatement_annual",
        "incomestatement_quarterly",
        "cashflow_annual",
        "cashflow_quarterly",
    ]
    first_kwargs = pipeline.azure_data_source.write_sql_table.call_args_list[0].kwargs
    assert set(first_kwargs["dtype_overrides"]) == {"DATE", "REPORT_DATE"}


def test_pull_financials_coerces_date_and_report_date_to_date():
    pipeline = FinancialData(tickers=["AAPL"])
    pipeline.pause_seconds = 0
    pipeline._pull_with_missing_ticker_retries = MagicMock(
        return_value=pd.DataFrame(
            {
                "TICKER": ["AAPL"],
                "DATE": [pd.Timestamp("2026-03-01 08:23:04")],
                "REPORT_DATE": [pd.Timestamp("2025-12-31 00:00:00")],
                "VALUE": [1.0],
            }
        )
    )

    out = pipeline._pull_financials(statement_type="financial", annual=True)

    assert str(out.loc[0, "DATE"]) == "2026-03-01"
    assert str(out.loc[0, "REPORT_DATE"]) == "2025-12-31"


def test_run_write_to_azure_skips_rows_duplicated_minus_date():
    pipeline = FinancialData(tickers=["AAPL"])
    pipeline.pause_seconds = 0
    incoming = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2025-01-02"],
            "ITEM": ["Revenue"],
            "VALUE": [100.0],
        }
    )
    existing = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2025-01-01"],
            "ITEM": ["Revenue"],
            "VALUE": [100.0],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=incoming)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=existing)
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True)

    pipeline.azure_data_source.write_sql_table.assert_not_called()
