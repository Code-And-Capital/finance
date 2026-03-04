from connectors.sqlite_data_source import SQLiteDataSource
import pytest
import tempfile
import os
import pandas as pd


def temp_db_file():
    """Helper to create a temporary SQLite file path."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    return path


def test_write_and_read_basic_table():
    source = SQLiteDataSource().set_database(temp_db_file())
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    source.write_sql_table("my_table", df, overwrite=True)
    result = source.read_sql_table(table_name="my_table")

    pd.testing.assert_frame_equal(result, df.astype(float))


def test_append_table_and_add_new_column():
    source = SQLiteDataSource().set_database(temp_db_file())
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5], "B": [6], "C": [7]})

    source.write_sql_table("tbl", df1, overwrite=True)
    source.write_sql_table("tbl", df2, overwrite=False)

    result = source.read_sql_table(table_name="tbl")

    assert set(result.columns) == {"A", "B", "C"}
    assert result.shape[0] == 3
    assert result.iloc[2]["C"] == 7.0
    assert pd.isna(result.iloc[0]["C"])


def test_delete_rows():
    source = SQLiteDataSource().set_database(temp_db_file())
    df = pd.DataFrame({"A": [1, 2, 3]})
    source.write_sql_table("tbl_del", df, overwrite=True)

    source.delete_sql_rows(table_name="tbl_del", where_clause="A=2")
    result = source.read_sql_table(table_name="tbl_del")

    assert result.shape[0] == 2
    assert 2 not in result["A"].values


def test_delete_rows_without_table_and_where_raises():
    source = SQLiteDataSource().set_database(temp_db_file())
    with pytest.raises(ValueError, match="table_name with a where_clause"):
        source.delete_sql_rows()


def test_delete_rows_without_where_raises():
    source = SQLiteDataSource().set_database(temp_db_file())
    with pytest.raises(ValueError, match="table_name with a where_clause"):
        source.delete_sql_rows(table_name="tbl")


def test_delete_rows_without_table_raises():
    source = SQLiteDataSource().set_database(temp_db_file())
    with pytest.raises(ValueError, match="table_name with a where_clause"):
        source.delete_sql_rows(where_clause="A=1")


def test_custom_query_read():
    source = SQLiteDataSource().set_database(temp_db_file())
    df = pd.DataFrame({"X": [10, 20], "Y": [30, 40]})
    source.write_sql_table("tbl_q", df, overwrite=True)

    result = source.read_sql_table(query="SELECT X FROM tbl_q")
    assert list(result.columns) == ["X"]
    assert result.iloc[0]["X"] == 10


def test_non_numeric_columns_remain_strings():
    source = SQLiteDataSource().set_database(temp_db_file())
    df = pd.DataFrame({"A": [1, 2], "B": ["foo", "bar"]})
    source.write_sql_table("tbl_str", df, overwrite=True)

    result = source.read_sql_table(table_name="tbl_str")
    assert result["A"].dtype == float
    assert result["B"].dtype == object
    assert result.iloc[0]["B"] == "foo"


def test_connect_and_disconnect_cycle():
    db_path = temp_db_file()
    source = SQLiteDataSource().connect(db_path)
    assert source.connection is not None
    source.disconnect()
    assert source.connection is None
