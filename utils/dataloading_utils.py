import pandas as pd
from bs4 import BeautifulSoup


def read_xls_file(file_path: str, sheet_number: int = 0, skiprows: int = 0):
    """
    Reads and formats an XML-based Excel file containing holdings data.

    This function:
    - Opens the XML file and parses it using BeautifulSoup.
    - Extracts a specific worksheet (by sheet_number) that contains the data.
    - Converts the worksheet's rows and cells into a pandas DataFrame.
    - Optionally skips initial rows (usually metadata or extra headers).
    - Uses the first actual data row as column headers and cleans them.

    Args:
        file_path (str): Path to the XML-based Excel file.
        sheet_number (int, optional): The index of the worksheet to extract (0-based). Defaults to 0.
        skiprows (int, optional): Number of rows to skip from the top of the worksheet. Defaults to 0.

    Returns:
        pd.DataFrame: A cleaned and formatted DataFrame containing the data from the specified worksheet.
    """
    # Open the XML file and parse it with BeautifulSoup
    with open(file_path) as xml_file:
        soup = BeautifulSoup(xml_file.read(), "xml")

        # Find the desired worksheet based on the provided sheet_number
        sheets = soup.find_all("Worksheet")[sheet_number]

        # Initialize a list to hold each row's data as a list of cell values
        sheet_as_list = []
        for row in sheets.find_all("Row"):
            # Extract text from each cell, use an empty string if no data
            sheet_as_list.append(
                [cell.Data.text if cell.Data else "" for cell in row.find_all("Cell")]
            )

        # Create a pandas DataFrame from the list of rows
        df = pd.DataFrame(sheet_as_list)

        # Remove the initial rows if specified by skiprows
        df = df.iloc[skiprows:]

        # Use the first row after skipping as column headers and drop that row
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])

        # Standardize column names: uppercase, replace spaces with underscores,
        # and remove unnecessary parts like "_(%)"
        df.columns = [
            col.upper().replace(" ", "_").replace("_(%)", "") for col in df.columns
        ]

    return df
