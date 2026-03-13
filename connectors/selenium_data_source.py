"""Stateful Selenium data source for downloading files from web pages."""

import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class SeleniumDataSource:
    """Stateful Selenium adapter for automated browser downloads."""

    def __init__(self, driver: webdriver.Chrome | None = None) -> None:
        """Initialize the data source with an optional pre-built driver."""
        self.driver: webdriver.Chrome | None = driver
        self.last_url: str | None = None

    @staticmethod
    def _build_chrome_options(download_folder: str) -> Options:
        """Build Chrome options configured for headless file downloads."""
        resolved_folder = Path(download_folder).expanduser().resolve()
        resolved_folder.mkdir(parents=True, exist_ok=True)

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
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": str(resolved_folder),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
            },
        )
        return options

    def download_data_file(
        self,
        *,
        url: str,
        download_folder: str,
        timeout_seconds: int = 15,
        wait_after_click_seconds: int = 3,
        selector: str = 'a.icon-xls-export[aria-label="Download data file"]',
    ) -> None:
        """Open a page, click the configured download link, and close the browser."""
        self.last_url = url
        self.driver = webdriver.Chrome(
            options=self._build_chrome_options(download_folder)
        )

        try:
            self.driver.get(url)
            download_link = WebDriverWait(self.driver, timeout_seconds).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            download_link.click()
            time.sleep(wait_after_click_seconds)
        finally:
            self.driver.quit()
            self.driver = None


default_selenium_data_source = SeleniumDataSource()

__all__ = ["SeleniumDataSource", "default_selenium_data_source"]
