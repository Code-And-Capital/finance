from unittest.mock import MagicMock

import pandas as pd

from pipelines.daily_market_data.financial_data import (
    EPSRevisionsData,
    EarningsSurprisesData,
    EstimatesData,
)


def test_eps_revisions_run_write_to_azure_writes_table():
    pipeline = EPSRevisionsData(tickers=["AAPL"])
    pulled = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "VALUE": [1.0]})
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    out = pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    pipeline.azure_data_source.get_engine.assert_called_once_with(
        configs_path="config/configs.json"
    )
    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "eps_revisions"
    assert set(kwargs["dtype_overrides"]) == {"DATE"}
    assert str(kwargs["df"]["DATE"].iloc[0]) == "2024-01-02"
    assert len(out) == 1


def test_eps_revisions_skip_write_when_duplicate_minus_date():
    pipeline = EPSRevisionsData(tickers=["AAPL"])
    pulled = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "VALUE": [1.0]})
    existing = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "FIGI": ["FIGI_AAPL"],
            "DATE": ["2024-01-01"],
            "VALUE": [1.0],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=existing)
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True, ticker_to_figi={"AAPL": "FIGI_AAPL"})

    pipeline.azure_data_source.write_sql_table.assert_not_called()


def test_earnings_surprises_run_write_to_azure_writes_table():
    pipeline = EarningsSurprisesData(tickers=["AAPL"])
    pulled = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "EARNINGS_DATE": ["2023-12-31"],
            "VALUE": [1.0],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    out = pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    pipeline.azure_data_source.get_engine.assert_called_once_with(
        configs_path="config/configs.json"
    )
    pipeline.azure_data_source.write_sql_table.assert_called_once()
    kwargs = pipeline.azure_data_source.write_sql_table.call_args.kwargs
    assert kwargs["table_name"] == "earnings_surprises"
    assert set(kwargs["dtype_overrides"]) == {"DATE", "EARNINGS_DATE"}
    assert str(kwargs["df"]["DATE"].iloc[0]) == "2024-01-02"
    assert str(kwargs["df"]["EARNINGS_DATE"].iloc[0]) == "2023-12-31"
    assert len(out) == 1


def test_earnings_surprises_skip_write_when_duplicate_minus_date():
    pipeline = EarningsSurprisesData(tickers=["AAPL"])
    pulled = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-02"],
            "EARNINGS_DATE": ["2023-12-31"],
            "VALUE": [1.0],
        }
    )
    existing = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "FIGI": ["FIGI_AAPL"],
            "DATE": ["2024-01-01"],
            "EARNINGS_DATE": [pd.Timestamp("2023-12-31")],
            "VALUE": [1.0],
        }
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(return_value=pulled)
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=existing)
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True, ticker_to_figi={"AAPL": "FIGI_AAPL"})

    pipeline.azure_data_source.write_sql_table.assert_not_called()


def test_estimates_run_returns_three_dataframes():
    pipeline = EstimatesData(tickers=["AAPL"])
    eps_df = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "VALUE": [1.0]})
    rev_df = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "VALUE": [2.0]})
    growth_df = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "VALUE": [3.0]}
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(
        side_effect=[eps_df, rev_df, growth_df]
    )

    out = pipeline.run()

    assert set(out.keys()) == {"eps", "revenue", "growth"}
    assert len(out["eps"]) == 1
    assert len(out["revenue"]) == 1
    assert len(out["growth"]) == 1


def test_estimates_run_write_to_azure_writes_three_tables():
    pipeline = EstimatesData(tickers=["AAPL"])
    eps_df = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "VALUE": [1.0]})
    rev_df = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "VALUE": [2.0]})
    growth_df = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "VALUE": [3.0]}
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(
        side_effect=[eps_df, rev_df, growth_df]
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(return_value=pd.DataFrame())
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True, configs_path="config/configs.json")

    pipeline.azure_data_source.get_engine.assert_called_once_with(
        configs_path="config/configs.json"
    )
    assert pipeline.azure_data_source.write_sql_table.call_count == 3
    table_names = [
        call.kwargs["table_name"]
        for call in pipeline.azure_data_source.write_sql_table.call_args_list
    ]
    assert table_names == ["eps_estimates", "revenue_estimates", "growth_estimates"]


def test_estimates_skip_write_when_duplicates_minus_date():
    pipeline = EstimatesData(tickers=["AAPL"])
    eps_df = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "VALUE": [1.0]})
    rev_df = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "VALUE": [2.0]})
    growth_df = pd.DataFrame(
        {"TICKER": ["AAPL"], "DATE": ["2024-01-02"], "VALUE": [3.0]}
    )
    pipeline._pull_with_missing_ticker_retries = MagicMock(
        side_effect=[eps_df, rev_df, growth_df]
    )
    pipeline.azure_data_source.get_engine = MagicMock(return_value=object())
    pipeline.azure_data_source.read_sql_table = MagicMock(
        side_effect=[
            pd.DataFrame(
                {
                    "TICKER": ["AAPL"],
                    "FIGI": ["FIGI_AAPL"],
                    "DATE": ["2024-01-01"],
                    "VALUE": [1.0],
                }
            ),
            pd.DataFrame(
                {
                    "TICKER": ["AAPL"],
                    "FIGI": ["FIGI_AAPL"],
                    "DATE": ["2024-01-01"],
                    "VALUE": [2.0],
                }
            ),
            pd.DataFrame(
                {
                    "TICKER": ["AAPL"],
                    "FIGI": ["FIGI_AAPL"],
                    "DATE": ["2024-01-01"],
                    "VALUE": [3.0],
                }
            ),
        ]
    )
    pipeline.azure_data_source.write_sql_table = MagicMock(return_value=None)

    pipeline.run(write_to_azure=True, ticker_to_figi={"AAPL": "FIGI_AAPL"})

    pipeline.azure_data_source.write_sql_table.assert_not_called()
