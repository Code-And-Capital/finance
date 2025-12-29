import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from utils.database_utils import find_missing_tickers


@patch("utils.database_utils.azure_utils.read_sql_table")
def test_find_missing_tickers_with_list(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame({"TICKER": ["AAPL", "MSFT", "GOOG"]})

    tickers = ["AAPL", "MSFT", "TSLA", "NFLX"]
    missing = find_missing_tickers("prices", tickers)

    assert set(missing) == {"TSLA", "NFLX"}


@patch("utils.database_utils.azure_utils.read_sql_table")
def test_find_missing_tickers_with_string(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame({"TICKER": ["AAPL", "MSFT", "GOOG"]})

    tickers = "TSLA"
    missing = find_missing_tickers("prices", tickers)

    assert missing == ["TSLA"]


@patch("utils.database_utils.azure_utils.read_sql_table")
def test_find_missing_tickers_with_numpy_array(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame({"TICKER": ["AAPL", "MSFT"]})

    tickers = np.array(["AAPL", "MSFT", "GOOG"])
    missing = find_missing_tickers("prices", tickers)

    assert missing == ["GOOG"]


@patch("utils.database_utils.azure_utils.read_sql_table")
def test_find_missing_tickers_all_present(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame({"TICKER": ["AAPL", "MSFT", "GOOG"]})

    tickers = ["AAPL", "GOOG"]
    missing = find_missing_tickers("prices", tickers)

    assert missing == []
