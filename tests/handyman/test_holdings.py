import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from handyman.holdings import get_index_holdings, get_llm_holdings


@patch("handyman.holdings.azure_utils.read_sql_table")
def test_get_index_holdings_with_string_inputs(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-01"],
            "INDEX": ["SP500"],
            "TICKER": ["AAPL"],
            "WEIGHT": [0.05],
        }
    )

    df = get_index_holdings(
        indices="SP500",
        tickers="AAPL",
        start_date="2024-01-01",
    )

    query = mock_read_sql.call_args.kwargs["query"]

    assert "SP500" in query
    assert "AAPL" in query
    assert "DATE >= '2024-01-01'" in query
    assert pd.api.types.is_datetime64_any_dtype(df["DATE"])


@patch("handyman.holdings.azure_utils.read_sql_table")
def test_get_index_holdings_with_list_inputs(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-01"],
            "INDEX": ["SP500", "NASDAQ100"],
            "TICKER": ["AAPL", "MSFT"],
        }
    )

    get_index_holdings(
        indices=["SP500", "NASDAQ100"],
        tickers=["AAPL", "MSFT"],
    )

    query = mock_read_sql.call_args.kwargs["query"]

    assert "SP500" in query
    assert "NASDAQ100" in query
    assert "AAPL" in query
    assert "MSFT" in query


@patch("handyman.holdings.azure_utils.read_sql_table")
def test_get_index_holdings_without_filters(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-01"],
            "INDEX": ["SP500"],
            "TICKER": ["AAPL"],
        }
    )

    df = get_index_holdings()

    query = mock_read_sql.call_args.kwargs["query"]

    assert "INDEX" not in query or "IN (" not in query
    assert "TICKER" not in query or "IN (" not in query
    assert "DATE >=" not in query
    assert len(df) == 1


@patch("handyman.holdings.azure_utils.read_sql_table")
def test_get_index_holdings_date_conversion(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-01"],
            "INDEX": ["SP500"],
            "TICKER": ["AAPL"],
        }
    )

    df = get_index_holdings(indices="SP500")

    assert pd.api.types.is_datetime64_any_dtype(df["DATE"])


def test_get_index_holdings_invalid_indices_type_raises():
    with pytest.raises(TypeError):
        get_index_holdings(indices=123)


def test_get_index_holdings_invalid_tickers_type_raises():
    with pytest.raises(TypeError):
        get_index_holdings(tickers=123)


@patch("handyman.holdings.azure_utils.read_sql_table")
def test_get_llm_holdings_with_list(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-02"],
            "strategy": ["LLM1", "LLM2"],
            "TICKER": ["AAPL", "MSFT"],
        }
    )

    llms = ["LLM1", "LLM3"]
    df = get_llm_holdings(llms=llms)

    query = mock_read_sql.call_args.kwargs["query"]
    assert "LLM1" in query
    assert "LLM3" in query
    assert isinstance(df, pd.DataFrame)
    assert set(df["strategy"]) == {"LLM1", "LLM2"}


@patch("handyman.holdings.azure_utils.read_sql_table")
def test_get_llm_holdings_with_string(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {"DATE": ["2024-01-01"], "strategy": ["LLM1"], "TICKER": ["AAPL"]}
    )

    df = get_llm_holdings(llms="LLM1")
    query = mock_read_sql.call_args.kwargs["query"]

    assert "LLM1" in query
    assert isinstance(df, pd.DataFrame)


@patch("handyman.holdings.azure_utils.read_sql_table")
def test_get_llm_holdings_with_array_like(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {
            "DATE": ["2024-01-01", "2024-01-02"],
            "strategy": ["LLM1", "LLM2"],
            "TICKER": ["AAPL", "MSFT"],
        }
    )

    llms = np.array(["LLM1", "LLM3"])
    df = get_llm_holdings(llms=llms)

    query = mock_read_sql.call_args.kwargs["query"]
    assert "LLM1" in query
    assert "LLM3" in query


@patch("handyman.holdings.azure_utils.read_sql_table")
def test_get_llm_holdings_with_start_date(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {"DATE": ["2024-01-01"], "strategy": ["LLM1"], "TICKER": ["AAPL"]}
    )

    df = get_llm_holdings(start_date="2024-01-01")
    query = mock_read_sql.call_args.kwargs["query"]

    assert "DATE >= '2024-01-01'" in query
    assert isinstance(df, pd.DataFrame)


@patch("handyman.holdings.azure_utils.read_sql_table")
def test_get_llm_holdings_no_filters(mock_read_sql):
    mock_read_sql.return_value = pd.DataFrame(
        {"DATE": ["2024-01-01"], "strategy": ["LLM1"], "TICKER": ["AAPL"]}
    )

    df = get_llm_holdings()
    query = mock_read_sql.call_args.kwargs["query"]

    assert "WHERE 1=1" in query
    assert isinstance(df, pd.DataFrame)
