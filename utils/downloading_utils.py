import utils.mapping as mapping
import utils.dataloading_utils as dataloading_utils
import utils.dataframe_utils as dataframe_utils
import os
from datetime import date

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def download_holdings(fund_name: str, url: str, download_folder: str = "."):
    """
    Downloads and processes holdings data for a specified fund from a given webpage.

    This function uses Selenium to:
    - Open the webpage for the fund's holdings.
    - Locate and click the download link for the holdings data file.
    - Wait for the download to complete.

    After downloading, it:
    - Reads the downloaded Excel file.
    - Filters the data to include only equity holdings from selected exchanges.
    - Cleans and normalizes the data.
    - Adds metadata such as the fund index and download date.

    Args:
        fund_name (str): The name of the fund to process (used for file mapping and metadata).
        url (str): The URL of the webpage containing the download link.
        download_folder (str, optional): The directory where the downloaded file will be saved. Defaults to the current directory.

    Returns:
        pd.DataFrame: A cleaned and processed DataFrame containing the fund's equity holdings.
    """
    # Configure Chrome options for headless operation and download directory
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-blink-features")
    options.add_argument("--disable-infobars")
    options.add_argument("--remote-debugging-port=9222")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    # options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    prefs = {"download.default_directory": download_folder}
    options.add_experimental_option("prefs", prefs)

    # Initialize the Selenium Chrome WebDriver
    driver = webdriver.Chrome(options=options)

    try:
        # Open the webpage containing the fund's holdings data
        driver.get(url)

        # Wait for the download link to be clickable and click it
        download_link = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, 'a.icon-xls-export[aria-label="Download data file"]')
            )
        )
        download_link.click()

        # Wait for the download to complete (basic wait, can be improved with polling the directory)
        time.sleep(3)
    finally:
        # Quit the WebDriver session to release resources
        driver.quit()

    # Load the downloaded Excel file using a helper function
    file_path = os.path.join(download_folder, mapping.etf_file_names[fund_name])
    df = dataloading_utils.read_xls_file(
        file_path=file_path, sheet_number=1, skiprows=7
    )

    # Filter for equity holdings only
    df = df[df["ASSET_CLASS"] == "Equity"]

    # Remove rows with missing or placeholder tickers
    df = df[df["TICKER"] != "--"]

    # Replace tickers according to the provided mapping
    df["TICKER"] = df["TICKER"].replace(mapping.ticker_mapping)

    # Filter to include only specific exchanges
    df = df[
        df["EXCHANGE"].isin(["NASDAQ", "New York Stock Exchange Inc.", "Nyse Mkt Llc"])
    ]

    # Convert relevant columns to numeric types where applicable
    df = dataframe_utils.convert_columns_to_numeric(df)

    # Remove entries with non-positive weights
    df = df[df["WEIGHT"] > 0]

    # Normalize the weight column to ensure it sums to 1
    df["WEIGHT"] = df["WEIGHT"] / df["WEIGHT"].sum()

    # Add metadata columns for the fund index and the current date
    df["INDEX"] = fund_name
    df["DATE"] = date.today()

    # Select and reorder relevant columns for the final output
    df = df[
        [
            "DATE",
            "INDEX",
            "TICKER",
            "NAME",
            "MARKET_VALUE",
            "WEIGHT",
            "QUANTITY",
            "PRICE",
            "LOCATION",
            "EXCHANGE",
            "CURRENCY",
            "FX_RATE",
        ]
    ]

    os.remove(file_path)
    return df
