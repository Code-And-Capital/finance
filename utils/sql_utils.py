import os
import pandas as pd
import sqlite3


def write_sql_table(
    database_name: str, table_name: str, df: pd.DataFrame, overwrite: bool = False
) -> None:
    """
    Write a pandas DataFrame to an SQLite table, adding new columns if necessary.

    Args:
        database_name (str): Name of the SQLite database (without extension).
        table_name (str): Table to write to.
        df (pd.DataFrame): DataFrame to write.
        overwrite (bool, optional): If True, overwrite the table. Defaults to False.
    """
    database_loc = os.path.join(
        os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath("__file__")), "..")
        ),
        "Data",
        f"{database_name}.sqlite",
    )

    conn = sqlite3.connect(database_loc)
    cursor = conn.cursor()

    if overwrite:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    else:
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (table_name,),
        )
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        else:
            # Get existing columns
            cursor.execute(f"PRAGMA table_info({table_name});")
            existing_cols = [col[1] for col in cursor.fetchall()]

            # Find new columns that need to be added
            new_cols = [col for col in df.columns if col not in existing_cols]

            for col in new_cols:
                # Assume TEXT type for new columns (safe default)
                cursor.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" TEXT;')
            conn.commit()

            # Append the DataFrame
            df.to_sql(table_name, conn, if_exists="append", index=False)

    conn.close()


def read_sql_table(
    database_name: str, table_name: str = None, query: str = None
) -> pd.DataFrame:
    """
    Reads data from an SQLite database into a pandas DataFrame.

    You can either:
    - Specify a table name to read all rows from that table.
    - Provide a custom SQL query for more flexible data retrieval.

    Args:
        database_name (str): The name of the SQLite database (without extension, e.g., 'holdings').
        table_name (str, optional): The name of the table to read. If provided, it overrides the `query`.
        query (str, optional): A custom SQL query. Ignored if `table_name` is provided.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the retrieved data.
    """
    # Construct the absolute path to the database file:
    # - Move to the parent directory of the script
    # - Enter the 'Data' folder
    # - Append the database name with '.sqlite' extension
    database_loc = os.path.join(
        os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath("__file__")), "..")
        ),
        "Data",
        f"{database_name}.sqlite",
    )

    # Connect to the SQLite database
    conn = sqlite3.connect(database=database_loc)

    # If a table name is provided, use a basic SELECT * query for that table
    if table_name:
        query = f"SELECT * FROM {table_name}"

    # Read the data from the database into a pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Close the database connection to ensure proper resource handling
    conn.close()

    # Attempt to convert each column to a numeric type where possible
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except (ValueError, TypeError):
            # If conversion fails (e.g., for string or mixed-type columns), leave as is
            pass

    return df
