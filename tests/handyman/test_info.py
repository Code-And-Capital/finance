import pandas as pd
import numpy as np
from unittest.mock import patch

from handyman.company_info import get_company_info


@patch("handyman.company_info.sql_utils.read_sql_table")
def test_get_company_info_with_string_ticker(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-01"],
            "sector": ["Tech"],
        }
    )

    df = get_company_info(tickers="AAPL")

    query = mock_read_sql.call_args.kwargs["query"]
    assert "AAPL" in query
    assert isinstance(df, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(df["DATE"])


@patch("handyman.company_info.sql_utils.read_sql_table")
def test_get_company_info_with_list_of_tickers(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "DATE": ["2024-01-01", "2024-01-02"],
            "sector": ["Tech", "Tech"],
        }
    )

    df = get_company_info(tickers=["AAPL", "MSFT"])

    query = mock_read_sql.call_args.kwargs["query"]
    assert "AAPL" in query
    assert "MSFT" in query
    assert len(df) == 2


@patch("handyman.company_info.sql_utils.read_sql_table")
def test_get_company_info_with_array_like_tickers(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-01"],
            "sector": ["Tech"],
        }
    )

    tickers = np.array(["AAPL"])
    df = get_company_info(tickers=tickers)

    query = mock_read_sql.call_args.kwargs["query"]
    assert "AAPL" in query
    assert isinstance(df, pd.DataFrame)


@patch("handyman.company_info.sql_utils.read_sql_table")
def test_get_company_info_with_start_date(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-05"],
            "sector": ["Tech"],
        }
    )

    df = get_company_info(start_date="2024-01-01")

    query = mock_read_sql.call_args.kwargs["query"]
    assert "DATE >= '2024-01-01'" in query
    assert pd.api.types.is_datetime64_any_dtype(df["DATE"])


@patch("handyman.company_info.sql_utils.read_sql_table")
def test_get_company_info_no_filters(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "DATE": ["2024-01-01"],
            "sector": ["Tech"],
        }
    )

    df = get_company_info()

    query = mock_read_sql.call_args.kwargs["query"]
    assert "WHERE 1=1" in query
    assert isinstance(df, pd.DataFrame)
