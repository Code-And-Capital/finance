"""Stateful XML/XLS data source implementation."""

from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


class XLSDataSource:
    """Stateful reader for XML-based XLS files."""

    def __init__(self, file_path: str | None = None) -> None:
        """Initialize XLS reader state.

        Parameters
        ----------
        file_path
            Optional default file path to load from.
        """
        self.file_path: str | None = file_path
        self.dataframe: pd.DataFrame | None = None

    def set_file_path(self, file_path: str) -> XLSDataSource:
        """Set the default source file path and return `self` for chaining."""
        self.file_path = file_path
        return self

    def read(self, sheet_number: int = 0, skiprows: int = 0) -> pd.DataFrame:
        """Read the currently configured file path into a DataFrame.

        The parsed DataFrame is stored on `self.dataframe` and returned.
        """
        if self.file_path is None:
            raise ValueError("No XLS file path is set. Call set_file_path() first.")

        path = Path(self.file_path)
        with path.open("r", encoding="utf-8") as xml_file:
            soup = BeautifulSoup(xml_file.read(), "xml")

        worksheets = soup.find_all("Worksheet")
        if sheet_number < 0 or sheet_number >= len(worksheets):
            raise IndexError(
                f"sheet_number {sheet_number} out of range for {len(worksheets)} sheets"
            )

        sheet = worksheets[sheet_number]
        rows = [
            [cell.Data.text if cell.Data else "" for cell in row.find_all("Cell")]
            for row in sheet.find_all("Row")
        ]

        dataframe = pd.DataFrame(rows).iloc[skiprows:]
        if dataframe.empty:
            self.dataframe = dataframe
            return dataframe

        dataframe = dataframe.rename(columns=dataframe.iloc[0]).drop(dataframe.index[0])
        dataframe.columns = [
            str(column).upper().replace(" ", "_").replace("_(%)", "")
            for column in dataframe.columns
        ]
        self.dataframe = dataframe
        return dataframe

    def read_xls_file(
        self,
        file_path: str,
        sheet_number: int = 0,
        skiprows: int = 0,
    ) -> pd.DataFrame:
        """Convenience method to set file path and read in one call."""
        return self.set_file_path(file_path).read(
            sheet_number=sheet_number, skiprows=skiprows
        )


default_xls_data_source = XLSDataSource()

__all__ = ["XLSDataSource", "default_xls_data_source"]
