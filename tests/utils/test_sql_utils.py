from utils.sql_utils import write_sql_table, read_sql_table, delete_sql_rows
import pytest
import tempfile
import pandas as pd


def temp_db_file():
    """Helper to create a temporary SQLite file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=True)
    return tmp.name


def test_write_and_read_basic_table():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    db_path = temp_db_file()

    # Write table
    write_sql_table(db_path, "my_table", df, overwrite=True)

    # Read it back
    result = read_sql_table(db_path, table_name="my_table")

    # Numeric columns converted to float
    pd.testing.assert_frame_equal(result, df.astype(float))


def test_append_table_and_add_new_column():
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5], "B": [6], "C": [7]})  # new column C
    db_path = temp_db_file()

    write_sql_table(db_path, "tbl", df1, overwrite=True)
    write_sql_table(db_path, "tbl", df2, overwrite=False)

    result = read_sql_table(db_path, table_name="tbl")

    # All columns should exist
    assert set(result.columns) == {"A", "B", "C"}
    # Three rows in total
    assert result.shape[0] == 3
    # Check new column values
    assert result.iloc[2]["C"] == 7.0
    # Existing rows have NaN for new column
    assert pd.isna(result.iloc[0]["C"])


def test_delete_rows():
    df = pd.DataFrame({"A": [1, 2, 3]})
    db_path = temp_db_file()
    write_sql_table(db_path, "tbl_del", df, overwrite=True)

    # Delete row where A == 2
    delete_sql_rows(db_path, table_name="tbl_del", where_clause="A=2")
    result = read_sql_table(db_path, table_name="tbl_del")

    assert result.shape[0] == 2
    assert 2 not in result["A"].values


def test_delete_rows_without_table_and_where_raises():
    db_path = temp_db_file()
    with pytest.raises(ValueError):
        delete_sql_rows(db_path)


def test_custom_query_read():
    df = pd.DataFrame({"X": [10, 20], "Y": [30, 40]})
    db_path = temp_db_file()
    write_sql_table(db_path, "tbl_q", df, overwrite=True)

    result = read_sql_table(db_path, query="SELECT X FROM tbl_q")
    assert list(result.columns) == ["X"]
    assert result.iloc[0]["X"] == 10


def test_non_numeric_columns_remain_strings():
    df = pd.DataFrame({"A": [1, 2], "B": ["foo", "bar"]})
    db_path = temp_db_file()
    write_sql_table(db_path, "tbl_str", df, overwrite=True)

    result = read_sql_table(db_path, "tbl_str")
    # Column A converted to float
    assert result["A"].dtype == float
    # Column B remains object/string
    assert result["B"].dtype == object
    assert result.iloc[0]["B"] == "foo"
