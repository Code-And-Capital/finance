import pandas as pd
import numpy as np
from decimal import Decimal
import logging
import pytest
import tempfile
from datetime import date
from unittest.mock import patch, MagicMock

from utils.dataframe_utils import (
    convert_columns_to_numeric,
    df_to_dict,
    normalize_columns,
)
from utils.math_utils import is_zero
from utils.logging import log
from utils.threading import ThreadWorkerPool
from utils.sql_utils import write_sql_table, read_sql_table, delete_sql_rows
import utils.downloading_utils as downloading_utils
from utils.dataloading_utils import read_xls_file


def test_all_numeric_columns():
    df = pd.DataFrame(
        {
            "A": ["1", "2", "3"],
            "B": [4, 5, 6],
        }
    )
    result = convert_columns_to_numeric(df.copy())

    assert result["A"].dtype.kind in "if"
    assert result["B"].dtype.kind in "if"
    assert result.equals(pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}))


def test_non_numeric_column_remains():
    df = pd.DataFrame(
        {
            "A": ["x", "y", "z"],
            "B": ["1", "2", "3"],
        }
    )
    result = convert_columns_to_numeric(df.copy())

    # B should convert
    assert result["B"].dtype.kind in "if"

    # A should remain unchanged
    assert result["A"].equals(df["A"])


def test_mixed_numeric_and_non_numeric_values():
    df = pd.DataFrame(
        {
            "A": ["1", "2", "bad"],
        }
    )
    result = convert_columns_to_numeric(df.copy())

    assert result["A"].dtype == object
    assert result["A"].equals(df["A"])


def test_numeric_with_commas_not_convertible():
    df = pd.DataFrame({"A": ["1,000", "2,000"]})

    # pd.to_numeric("1,000") raises ValueError, so this column should remain unchanged
    result = convert_columns_to_numeric(df.copy())

    assert result["A"].tolist() == ["1,000", "2,000"]
    assert result["A"].dtype == object


def test_column_with_none_values():
    df = pd.DataFrame({"A": [1, None, "3"]})
    result = convert_columns_to_numeric(df.copy())

    assert result["A"].dtype.kind in "if"


def test_empty_dataframe():
    df = pd.DataFrame()
    result = convert_columns_to_numeric(df.copy())
    assert result.empty
    assert list(result.columns) == []


def test_mixed_column_types():
    df = pd.DataFrame(
        {
            "A": ["1", 2, 3.0],
            "B": ["x", 1, 2],
        }
    )
    result = convert_columns_to_numeric(df.copy())

    # A fully numeric
    assert result["A"].dtype.kind in "if"

    # B should coerce x to NaN
    assert result["B"].equals(df["B"])


def test_basic_conversion():
    df = pd.DataFrame({"key": ["A", "B", "C"], "value": [1, 2, 3]})
    result = df_to_dict(df, "key", "value")
    assert result == {"A": 1, "B": 2, "C": 3}


def test_duplicate_keys_overwrite():
    df = pd.DataFrame({"key": ["A", "A", "B"], "value": [1, 2, 3]})
    result = df_to_dict(df, "key", "value")

    # Pandas set_index keeps the last occurrence as the dict value
    assert result == {"A": 2, "B": 3}


def test_missing_key_column():
    df = pd.DataFrame({"value": [1, 2]})
    with pytest.raises(KeyError):
        df_to_dict(df, "missing", "value")


def test_missing_value_column():
    df = pd.DataFrame({"key": ["A", "B"]})
    with pytest.raises(KeyError):
        df_to_dict(df, "key", "missing")


def test_null_keys_allowed():
    df = pd.DataFrame({"key": ["A", None, "C"], "value": [1, 2, 3]})
    result = df_to_dict(df, "key", "value")

    # None becomes a valid dict key
    assert result == {"A": 1, None: 2, "C": 3}


def test_null_values():
    df = pd.DataFrame({"key": ["A", "B", "C"], "value": [1, None, 3]})
    result = df_to_dict(df, "key", "value")

    assert result["A"] == 1
    assert np.isnan(result["B"])
    assert result["C"] == 3
    assert set(result.keys()) == {"A", "B", "C"}


def test_mixed_type_keys():
    df = pd.DataFrame({"key": ["A", 100, 3.14], "value": [1, 2, 3]})
    result = df_to_dict(df, "key", "value")
    assert result == {"A": 1, 100: 2, 3.14: 3}


def test_empty_dataframe():
    df = pd.DataFrame(columns=["key", "value"])
    result = df_to_dict(df, "key", "value")
    assert result == {}


def test_non_string_column_names():
    df = pd.DataFrame({0: ["A", "B"], 1: [10, 20]})
    result = df_to_dict(df, 0, 1)
    assert result == {"A": 10, "B": 20}


def test_multiindex_columns_still_work():
    df = pd.DataFrame({("col", "key"): ["A", "B"], ("col", "value"): [1, 2]})
    result = df_to_dict(df, ("col", "key"), ("col", "value"))
    assert result == {"A": 1, "B": 2}


def test_basic_normalization():
    df = pd.DataFrame(columns=["col one", "colTwo", "COL three"])
    result = normalize_columns(df.copy())

    assert result.columns.tolist() == ["COL_ONE", "COLTWO", "COL_THREE"]


def test_no_spaces_only_uppercase():
    df = pd.DataFrame(columns=["abc", "Def", "GHI"])
    result = normalize_columns(df.copy())

    assert result.columns.tolist() == ["ABC", "DEF", "GHI"]


def test_special_characters_remain():
    df = pd.DataFrame(columns=["col-name", "col/name", "col.name"])
    result = normalize_columns(df.copy())

    assert result.columns.tolist() == ["COL-NAME", "COL/NAME", "COL.NAME"]


def test_numeric_column_names():
    df = pd.DataFrame(columns=[1, 2, 3])
    result = normalize_columns(df.copy())

    # Numeric names converted to strings then uppercased (no change)
    assert result.columns.tolist() == ["1", "2", "3"]


def test_mixed_types_in_columns():
    df = pd.DataFrame(columns=["name", 123, None])
    result = normalize_columns(df.copy())

    # None becomes "NONE", numbers become strings
    assert result.columns.tolist() == ["NAME", "123", "NONE"]


def test_duplicate_columns_after_normalization():
    df = pd.DataFrame(columns=["col one", "COL_ONE"])
    result = normalize_columns(df.copy())

    # Pandas will allow duplicate column names
    assert result.columns.tolist() == ["COL_ONE", "COL_ONE"]


def test_empty_dataframe():
    df = pd.DataFrame()
    result = normalize_columns(df.copy())

    assert list(result.columns) == []


def test_leading_trailing_spaces():
    df = pd.DataFrame(columns=["  col  one  ", " col two"])
    result = normalize_columns(df.copy())

    # Leading/trailing spaces become underscores as well
    assert result.columns.tolist() == ["__COL__ONE__", "_COL_TWO"]


def test_exact_zero():
    assert is_zero(0.0) is True
    assert is_zero(0) is True


def test_small_positive_within_tol():
    assert is_zero(1e-15) is True
    assert is_zero(5e-13, tol=1e-12) is True


def test_small_negative_within_tol():
    assert is_zero(-5e-13, tol=1e-12) is True


def test_exactly_equal_to_tol():
    # abs(x) < tol → equal-to-tolerance should be False
    assert is_zero(1e-12) is False
    assert is_zero(-1e-12) is False


def test_just_outside_tol():
    assert is_zero(1.0000001e-12) is False
    assert is_zero(-1.0000001e-12) is False


def test_larger_values():
    assert is_zero(1e-6) is False
    assert is_zero(-1e-6) is False


def test_non_float_numeric_types():
    assert is_zero(np.float64(0.0))
    assert is_zero(np.float64(1e-15))

    assert is_zero(Decimal("1e-13"), tol=Decimal("1e-12"))
    assert not is_zero(Decimal("1e-4"), tol=Decimal("1e-12"))


def test_custom_tolerance():
    assert is_zero(0.01, tol=0.1) is True
    assert is_zero(0.01, tol=0.001) is False


def test_non_numeric_raises():
    with pytest.raises(TypeError):
        is_zero("0")

    with pytest.raises(TypeError):
        is_zero(None)


def test_info_log(caplog):
    with caplog.at_level(logging.DEBUG):
        log("hello", type="info")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.INFO
    assert record.message == "hello"


def test_warning_log(caplog):
    with caplog.at_level(logging.DEBUG):
        log("warn msg", type="warning")

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert caplog.records[0].message == "warn msg"


def test_error_log(caplog):
    with caplog.at_level(logging.DEBUG):
        log("err msg", type="error")

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR


def test_debug_log(caplog):
    with caplog.at_level(logging.DEBUG):
        log("dbg msg", type="debug")

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG


def test_critical_log(caplog):
    with caplog.at_level(logging.DEBUG):
        log("crit msg", type="critical")

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.CRITICAL


def test_invalid_type_defaults_to_info(caplog):
    with caplog.at_level(logging.DEBUG):
        log("fallback", type="notalevel")

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    assert caplog.records[0].message == "fallback"


def test_custom_logger_name(caplog):
    with caplog.at_level(logging.DEBUG):
        log("msg", name="custom_logger")

    assert len(caplog.records) == 1
    assert caplog.records[0].name == "custom_logger"


def test_handler_added_only_once(caplog):
    # get the logger explicitly so we can inspect handlers
    logger = logging.getLogger("handler_test_logger")
    logger.handlers.clear()

    with caplog.at_level(logging.DEBUG):
        log("first", name="handler_test_logger")
        log("second", name="handler_test_logger")

    # Only one handler should still exist
    assert len(logger.handlers) == 1

    # Two log records should be emitted
    assert len(caplog.records) == 2
    assert caplog.records[0].message == "first"
    assert caplog.records[1].message == "second"


def test_logger_level_is_debug(caplog):
    # ensures logger is always set to DEBUG regardless of input type
    logger = logging.getLogger("levelcheck_logger")
    logger.handlers.clear()

    with caplog.at_level(logging.DEBUG):
        log("test", name="levelcheck_logger")

    assert logger.level == logging.DEBUG


def test_run_successful_tasks():
    pool = ThreadWorkerPool(max_workers=4)

    def task1():
        return 1

    def task2():
        return 2

    def task3():
        return 3

    tasks = [task1, task2, task3]
    results = pool.run(tasks)

    # Results returned in order
    assert results == [1, 2, 3]


def test_run_tasks_out_of_order_completion():
    pool = ThreadWorkerPool(max_workers=3)

    def task_fast():
        return "fast"

    def task_slow():
        import time

        time.sleep(0.1)
        return "slow"

    tasks = [task_slow, task_fast, task_slow]
    results = pool.run(tasks)

    # Order must match input, not completion order
    assert results[0] == "slow"
    assert results[1] == "fast"
    assert results[2] == "slow"


def test_run_tasks_with_exceptions():
    pool = ThreadWorkerPool(max_workers=2)

    def task_ok():
        return 42

    def task_fail():
        raise ValueError("boom")

    tasks = [task_ok, task_fail, task_ok]

    with patch("utils.logging.log") as mock_log:
        results = pool.run(tasks)

    # The failed task should return None
    assert results == [42, None, 42]

    # Logging called once with an error
    mock_log.assert_called_once()
    args, kwargs = mock_log.call_args
    assert "Task 1 failed" in args[0]
    assert kwargs["type"] == "error"


def test_run_empty_task_list():
    pool = ThreadWorkerPool(max_workers=2)
    results = pool.run([])
    assert results == []


def test_run_with_max_workers_parameter():
    # Basic sanity check that setting max_workers doesn't break functionality
    pool = ThreadWorkerPool(max_workers=1)

    def task():
        return "ok"

    results = pool.run([task for _ in range(3)])
    assert results == ["ok", "ok", "ok"]


def test_run_task_with_sleep_to_simulate_delay():
    pool = ThreadWorkerPool(max_workers=3)

    def task1():
        return 1

    def task2():
        import time

        time.sleep(0.05)
        return 2

    def task3():
        return 3

    tasks = [task1, task2, task3]
    results = pool.run(tasks)
    assert results == [1, 2, 3]


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


def test_read_xls_file_parses_xml_correctly(tmp_path):
    # --- Arrange ---
    xml_content = """
    <Workbook>
        <Worksheet>
            <Table>
                <Row>
                    <Cell><Data>Ignore1</Data></Cell>
                </Row>
                <Row>
                    <Cell><Data>Col A</Data></Cell>
                    <Cell><Data>Col B (%)</Data></Cell>
                </Row>
                <Row>
                    <Cell><Data>1</Data></Cell>
                    <Cell><Data>2</Data></Cell>
                </Row>
            </Table>
        </Worksheet>

        <Worksheet>
            <Table>
                <Row>
                    <Cell><Data>X</Data></Cell>
                    <Cell><Data>Y</Data></Cell>
                </Row>
                <Row>
                    <Cell><Data>9</Data></Cell>
                    <Cell><Data>8</Data></Cell>
                </Row>
            </Table>
        </Worksheet>
    </Workbook>
    """

    file_path = tmp_path / "testfile.xml"
    file_path.write_text(xml_content)

    # --- Act ---
    df = read_xls_file(file_path=str(file_path), sheet_number=0, skiprows=1)

    # --- Assert ---
    # Expected columns after normalization:
    # "COL A"  -> "COL_A"
    # "COL B (%)" -> "COL_B"
    assert list(df.columns) == ["COL_A", "COL_B"]

    # Expected data
    assert df.iloc[0]["COL_A"] == "1"
    assert df.iloc[0]["COL_B"] == "2"


def test_read_xls_file_selects_correct_sheet(tmp_path):
    # --- Arrange ---
    xml_content = """
    <Workbook>
        <Worksheet>
            <Table>
                <Row><Cell><Data>A</Data></Cell></Row>
                <Row><Cell><Data>1</Data></Cell></Row>
            </Table>
        </Worksheet>

        <Worksheet>
            <Table>
                <Row><Cell><Data>Header1</Data></Cell></Row>
                <Row><Cell><Data>Value1</Data></Cell></Row>
            </Table>
        </Worksheet>
    </Workbook>
    """

    file_path = tmp_path / "testfile.xml"
    file_path.write_text(xml_content)

    # --- Act ---
    df = read_xls_file(str(file_path), sheet_number=1, skiprows=0)

    # --- Assert ---
    assert list(df.columns) == ["HEADER1"]
    assert df.iloc[0]["HEADER1"] == "Value1"
