import pandas as pd
import numpy as np
from datetime import date
import pytest
from unittest.mock import patch, MagicMock

from utils.yfinance_utils import (
    pull_prices,
    pull_financials,
    create_client,
    pull_info,
    pull_officers,
)


@patch("utils.yfinance_utils.azure_utils.read_sql_table")
@patch("utils.yfinance_utils.add_missing_tickers")
@patch("utils.yfinance_utils.dataframe_utils.df_to_dict")
@patch("utils.yfinance_utils.yahoo_finance.YahooDataClient")
def test_pull_prices_happy_path(
    mock_client_class, mock_df_to_dict, mock_add_missing, mock_read_sql
):
    # Mock SQL return
    mock_read_sql.return_value = pd.DataFrame(
        {"TICKER": ["AAPL", "MSFT"], "START_DATE": ["2024-01-01", "2024-01-02"]}
    )

    # Mock add_missing_tickers returns same df
    mock_add_missing.side_effect = lambda df, tickers: df

    # Mock df_to_dict returns predictable mapping
    mock_df_to_dict.return_value = {
        "AAPL": pd.Timestamp("2024-01-02"),
        "MSFT": pd.Timestamp("2024-01-03"),
    }

    # Mock Yahoo client
    mock_client = MagicMock()
    mock_client.get_prices.return_value = pd.DataFrame(
        {"DATE": ["2024-01-02", "2024-01-03"], "AAPL": [100, 101], "MSFT": [200, 201]}
    )
    mock_client_class.return_value = mock_client

    result = pull_prices(["AAPL", "MSFT"], client=None)

    # Validate call chain
    mock_read_sql.assert_called_once()
    mock_add_missing.assert_called_once()
    mock_df_to_dict.assert_called_once()
    mock_client.get_prices.assert_called_once()

    # Validate result type and content
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) >= {"DATE", "AAPL", "MSFT"}
    assert result.loc[0, "AAPL"] == 100


@patch("utils.yfinance_utils.azure_utils.read_sql_table")
@patch("utils.yfinance_utils.add_missing_tickers")
@patch("utils.yfinance_utils.dataframe_utils.df_to_dict")
def test_pull_prices_with_array_like_input(
    mock_df_to_dict, mock_add_missing, mock_read_sql
):
    # Simulate a numpy array from df["TICKER"].unique()
    import numpy as np

    tickers = np.array(["AAPL", "MSFT"])

    mock_read_sql.return_value = pd.DataFrame(
        {"TICKER": ["AAPL", "MSFT"], "START_DATE": ["2024-01-01", "2024-01-02"]}
    )

    mock_add_missing.side_effect = lambda df, tickers: df
    mock_df_to_dict.return_value = {
        "AAPL": pd.Timestamp("2024-01-02"),
        "MSFT": pd.Timestamp("2024-01-03"),
    }

    # Provide a dummy client to avoid Yahoo API calls
    dummy_client = MagicMock()
    dummy_client.get_prices.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "DATE": ["2024-01-02", "2024-01-03"],
            "ADJ_CLOSE": [150.0, 300.0],
        }
    )

    result = pull_prices(tickers, client=dummy_client)
    assert isinstance(result, pd.DataFrame)


@patch("utils.yfinance_utils.yahoo_finance.YahooDataClient")
def test_create_client_with_string(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    client = create_client("AAPL")

    mock_client_class.assert_called_once_with(
        ["AAPL"],
        max_workers=10,
    )
    assert client is mock_client


@patch("utils.yfinance_utils.yahoo_finance.YahooDataClient")
def test_create_client_with_list(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    tickers = ["AAPL", "MSFT"]
    client = create_client(tickers, max_workers=5)

    mock_client_class.assert_called_once_with(
        ["AAPL", "MSFT"],
        max_workers=5,
    )
    assert client is mock_client


@patch("utils.yfinance_utils.yahoo_finance.YahooDataClient")
def test_create_client_with_array_like(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    tickers = np.array(["AAPL", "MSFT"])
    client = create_client(tickers)

    mock_client_class.assert_called_once_with(
        ["AAPL", "MSFT"],
        max_workers=10,
    )
    assert client is mock_client


def make_mock_client_fin():
    client = MagicMock()
    client.get_financials.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "DATE": ["2024-01-02", "2024-01-02"],
            "REPORT_DATE": ["2023-12-31", "2023-12-31"],
            "metric": ["revenue", "revenue"],
            "value": [100, 200],
        }
    )
    return client


def test_pull_financials_with_string_ticker():
    client = make_mock_client_fin()

    df = pull_financials(
        tickers="AAPL",
        client=client,
    )

    client.get_financials.assert_called_once()
    assert isinstance(df, pd.DataFrame)
    assert "TICKER" in df.columns


def test_pull_financials_with_list_tickers():
    client = make_mock_client_fin()

    df = pull_financials(
        tickers=["AAPL", "MSFT"],
        annual=False,
        statement_type="income_statement",
        client=client,
    )

    client.get_financials.assert_called_once_with(
        annual=False,
        statement_type="income_statement",
    )
    assert len(df) == 2


def test_pull_financials_with_array_like_tickers():
    client = make_mock_client_fin()

    tickers = np.array(["AAPL", "MSFT"])
    df = pull_financials(tickers=tickers, client=client)

    assert isinstance(df, pd.DataFrame)


def test_pull_financials_invalid_statement_type_raises():
    client = make_mock_client_fin()

    with pytest.raises(ValueError) as excinfo:
        pull_financials(
            tickers="AAPL",
            statement_type="invalid_statement",
            client=client,
        )

    assert "statement_type must be one of" in str(excinfo.value)


@patch("utils.yfinance_utils.yahoo_finance.YahooDataClient")
def test_pull_financials_creates_client_if_missing(mock_client_class):
    mock_client = MagicMock()
    mock_client.get_financials.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "DATE": ["2024-01-02", "2024-01-02"],
            "REPORT_DATE": ["2023-12-31", "2023-12-31"],
            "metric": ["revenue", "revenue"],
            "value": [100, 200],
        }
    )
    mock_client_class.return_value = mock_client

    df = pull_financials(tickers="AAPL")

    mock_client_class.assert_called_once()
    mock_client.get_financials.assert_called_once()
    assert isinstance(df, pd.DataFrame)


def make_mock_client_info():
    client = MagicMock()
    client.get_company_info.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "DATE": ["2024-01-02", "2024-01-02"],
            "employees": [100000, 200000],
            "sector": ["Tech", "Tech"],
        }
    )
    return client


def test_pull_info_with_injected_client():
    client = make_mock_client_info()

    df = pull_info(tickers="AAPL", client=client)

    client.get_company_info.assert_called_once()
    assert isinstance(df, pd.DataFrame)

    assert df.dtypes.iloc[0] == object


@patch("utils.yfinance_utils.create_client")
def test_pull_info_creates_client_if_missing(mock_create_client):
    client = make_mock_client_info()
    mock_create_client.return_value = client

    df = pull_info(tickers=["AAPL", "MSFT"])

    mock_create_client.assert_called_once_with(tickers=["AAPL", "MSFT"])
    client.get_company_info.assert_called_once()
    assert isinstance(df, pd.DataFrame)


def test_pull_info_list_to_nan():
    # Mock client with get_company_info returning a controlled DataFrame
    mock_client = MagicMock()
    mock_client.get_company_info.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT", "GOOGL"],
            "DATE": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "VALUE": [100, [1, 2, 3], "text"],
        }
    )

    result = pull_info(tickers=["AAPL", "MSFT", "GOOGL"], client=mock_client)

    # Check that the DATE column was converted to date objects
    assert pd.api.types.is_object_dtype(result["DATE"])
    assert isinstance(result["DATE"].iloc[0], date)

    # Check that the list in VALUE column became np.nan
    assert np.isnan(result["VALUE"].iloc[1])

    # Other values remain unchanged
    assert result["VALUE"].iloc[0] == 100
    assert result["VALUE"].iloc[2] == "text"


def make_mock_client_officer():
    client = MagicMock()
    client.get_officer_info.return_value = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "DATE": ["2024-01-02", "2024-01-02"],
            "name": ["Tim Cook", "Satya Nadella"],
            "title": ["CEO", "CEO"],
        }
    )
    return client


def test_pull_officers_with_injected_client():
    client = make_mock_client_officer()

    df = pull_officers(tickers="AAPL", client=client)

    client.get_officer_info.assert_called_once()
    assert isinstance(df, pd.DataFrame)
    assert "name" in df.columns


@patch("utils.yfinance_utils.create_client")
def test_pull_officers_creates_client_if_missing(mock_create_client):
    client = make_mock_client_officer()
    mock_create_client.return_value = client

    df = pull_officers(tickers=["AAPL", "MSFT"])

    mock_create_client.assert_called_once_with(tickers=["AAPL", "MSFT"])
    client.get_officer_info.assert_called_once()
    assert isinstance(df, pd.DataFrame)
