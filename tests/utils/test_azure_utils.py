import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.engine import Engine, URL

import utils.azure_utils as azure_utils


# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def fake_config(tmp_path):
    cfg = {
        "azure": {
            "server": "fake_server",
            "database": "fake_db",
            "username": "user",
            "password": "pass",
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture
def mock_engine():
    return MagicMock(spec=Engine)


# -----------------------------
# get_azure_engine
# -----------------------------


def test_get_azure_engine_creates_engine(fake_config, monkeypatch):
    fake_created = MagicMock()
    monkeypatch.setattr(
        azure_utils, "create_engine", lambda url, **kwargs: fake_created
    )

    engine = azure_utils.get_azure_engine(str(fake_config), driver="Fake Driver")

    assert engine is fake_created
    # URL object is created properly
    url_obj = azure_utils.URL.create(
        "mssql+pyodbc",
        query={
            "odbc_connect": f"Driver={{Fake Driver}};Server=tcp:fake_server,1433;Database=fake_db;Uid=user;Pwd=pass;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        },
    )
    assert url_obj.drivername == "mssql+pyodbc"


def test_get_azure_engine_missing_azure_key(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError):
        azure_utils.get_azure_engine(str(path))


# -----------------------------
# _infer_sqlserver_type
# -----------------------------


@pytest.mark.parametrize(
    "series,expected",
    [
        (pd.Series([1, 2, 3]), "BIGINT"),
        (pd.Series([1.1, 2.2]), "FLOAT"),
        (pd.Series([True, False]), "BIT"),
        (pd.Series(pd.date_range("2020-01-01", periods=2)), "DATETIME2"),
        (pd.Series(["a", "b"]), "NVARCHAR(MAX)"),
    ],
)
def test_infer_sqlserver_type(series, expected):
    assert azure_utils._infer_sqlserver_type(series) == expected


# -----------------------------
# _quote_ident
# -----------------------------


def test_quote_ident_basic():
    assert azure_utils._quote_ident("col") == "[col]"


def test_quote_ident_escapes_bracket():
    assert azure_utils._quote_ident("weird]name") == "[weird]]name]"


def test_quote_ident_none():
    import pytest

    with pytest.raises(ValueError):
        azure_utils._quote_ident(None)


# -----------------------------
# write_sql_table (mocked)
# -----------------------------


def test_write_sql_table_overwrite(mock_engine):
    df = pd.DataFrame({"a": [1, 2]})

    # Patch df.to_sql to confirm call
    with patch("pandas.DataFrame.to_sql") as mock_to_sql:
        azure_utils.write_sql_table(mock_engine, "table", df, overwrite=True)
        mock_to_sql.assert_called_once()
        args, kwargs = mock_to_sql.call_args
        assert kwargs["if_exists"] == "replace"
        assert kwargs["index"] is False


def test_write_sql_table_append(mock_engine):
    df = pd.DataFrame({"a": [1, 2]})
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    # Simulate table exists with one column 'a'
    mock_conn.execute.return_value.scalar.return_value = 1
    mock_conn.execute.return_value.fetchall.return_value = [("a",)]

    with patch("pandas.DataFrame.to_sql") as mock_to_sql:
        azure_utils.write_sql_table(mock_engine, "table", df, overwrite=False)
        mock_to_sql.assert_called_once()
        args, kwargs = mock_to_sql.call_args
        assert kwargs["if_exists"] == "append"


# -----------------------------
# read_sql_table (mocked)
# -----------------------------


def test_read_sql_table_by_name(mock_engine):
    fake_df = pd.DataFrame({"x": [1, 2]})
    with patch("pandas.read_sql_query", return_value=fake_df) as mock_read:
        out = azure_utils.read_sql_table(mock_engine, table_name="tbl")
        mock_read.assert_called_once()
        assert out.equals(fake_df)


def test_read_sql_table_requires_input(mock_engine):
    import pytest

    with pytest.raises(ValueError):
        azure_utils.read_sql_table(mock_engine)


# -----------------------------
# delete_sql_rows (mocked)
# -----------------------------


def test_delete_sql_rows_executes(mock_engine):
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    azure_utils.delete_sql_rows(mock_engine, "table", "x>1")
    mock_conn.exec_driver_sql.assert_called_once()
    sql = mock_conn.exec_driver_sql.call_args[0][0]
    assert "DELETE FROM" in sql


def test_delete_sql_rows_requires_where(mock_engine):
    import pytest

    with pytest.raises(ValueError):
        azure_utils.delete_sql_rows(mock_engine, "table", "   ")


# -----------------------------
# execute_sql (mocked)
# -----------------------------


def test_execute_sql_executes(mock_engine):
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    azure_utils.execute_sql(mock_engine, "CREATE TABLE x(y int)")
    mock_conn.exec_driver_sql.assert_called_once()
