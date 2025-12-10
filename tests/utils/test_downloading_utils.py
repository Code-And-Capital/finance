import utils.downloading_utils as downloading_utils
from datetime import date
import pandas as pd
from unittest.mock import MagicMock


def test_download_holdings_full_processing(monkeypatch):
    # -------------------------
    # Mock Selenium
    # -------------------------
    mock_driver = MagicMock()
    mock_driver.get.return_value = None
    mock_driver.quit.return_value = None
    monkeypatch.setattr(
        downloading_utils.webdriver, "Chrome", lambda options=None: mock_driver
    )

    mock_wait = MagicMock()
    mock_wait.until.return_value = MagicMock(click=lambda: None)
    monkeypatch.setattr(
        downloading_utils, "WebDriverWait", lambda driver, timeout: mock_wait
    )

    # -------------------------
    # Mock read_xls_file
    # -------------------------
    fake_raw_df = pd.DataFrame(
        {
            "ASSET_CLASS": ["Equity", "Equity", "Bond", "Equity"],
            "TICKER": ["AAPL", "--", "MSFT", "BAD"],
            "NAME": ["Apple", "BadRow", "Microsoft", "BadCorp"],
            "MARKET_VALUE": ["100", "999", "200", "300"],
            "WEIGHT": ["0.4", "0.1", "-0.2", "0.6"],
            "QUANTITY": ["10", "999", "5", "20"],
            "PRICE": ["10", "1", "40", "15"],
            "LOCATION": ["USA", "USA", "USA", "USA"],
            "EXCHANGE": [
                "NASDAQ",
                "New York Stock Exchange Inc.",
                "Nyse Mkt Llc",
                "InvalidExchange",
            ],
            "CURRENCY": ["USD", "USD", "USD", "USD"],
            "FX_RATE": ["1.0", "1.0", "1.0", "1.0"],
        }
    )

    mock_read_xls = MagicMock(return_value=fake_raw_df)
    monkeypatch.setattr(
        downloading_utils.dataloading_utils, "read_xls_file", mock_read_xls
    )

    # -------------------------
    # Mock mapping
    # -------------------------
    monkeypatch.setattr(
        downloading_utils.mapping, "etf_file_names", {"TESTFUND": "fakefile.xlsx"}
    )
    monkeypatch.setattr(downloading_utils.mapping, "ticker_mapping", {"BAD": "GOOD"})

    # -------------------------
    # Mock os.remove to simulate deletion
    # -------------------------
    monkeypatch.setattr(downloading_utils.os, "remove", lambda path: None)

    # -------------------------
    # Act
    # -------------------------
    result = downloading_utils.download_holdings(
        fund_name="TESTFUND", url="http://dummy", download_folder="/tmp"
    )

    # -------------------------
    # Assert: processing logic
    # -------------------------
    assert len(result) == 1
    row = result.iloc[0]
    assert row["TICKER"] == "AAPL"
    assert row["EXCHANGE"] == "NASDAQ"
    assert abs(row["WEIGHT"] - 1.0) < 1e-6
    assert row["INDEX"] == "TESTFUND"
    assert row["DATE"] == date.today()
