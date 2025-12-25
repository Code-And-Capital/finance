import pandas as pd
from unittest.mock import patch

from handyman.prices import get_prices


@patch("handyman.prices.sql_utils.read_sql_table")
def test_get_prices_returns_pivoted_dataframe(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "TICKER": ["AAPL", "MSFT", "AAPL"],
            "ADJ_CLOSE": [100.0, 200.0, 101.0],
        }
    )

    result = get_prices(
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-01",
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["AAPL", "MSFT"]
    assert result.index.name == "DATE"

    assert result.loc[pd.Timestamp("2024-01-01"), "AAPL"] == 100.0
    assert result.loc[pd.Timestamp("2024-01-01"), "MSFT"] == 200.0
    assert result.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 101.0


@patch("handyman.prices.sql_utils.read_sql_table")
def test_get_prices_converts_date_to_datetime(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-01"],
            "TICKER": ["AAPL"],
            "ADJ_CLOSE": [100.0],
        }
    )

    result = get_prices(["AAPL"], "2024-01-01")

    assert isinstance(result.index, pd.DatetimeIndex)


@patch("handyman.prices.sql_utils.read_sql_table")
def test_get_prices_builds_correct_sql_query(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-01"],
            "TICKER": ["AAPL"],
            "ADJ_CLOSE": [100.0],
        }
    )

    get_prices(["AAPL", "MSFT"], "2024-01-01")

    query = mock_read_sql.call_args.kwargs["query"]

    assert "AAPL" in query
    assert "MSFT" in query
    assert "DATE >= '2024-01-01'" in query


@patch("handyman.prices.sql_utils.read_sql_table")
def test_get_prices_missing_values_are_nan(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "TICKER": ["AAPL", "MSFT", "AAPL"],
            "ADJ_CLOSE": [100.0, 200.0, 101.0],
        }
    )

    result = get_prices(["AAPL", "MSFT"], "2024-01-01")

    assert pd.isna(result.loc[pd.Timestamp("2024-01-02"), "MSFT"])
