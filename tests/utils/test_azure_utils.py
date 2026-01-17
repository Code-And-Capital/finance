import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.engine import Engine

import utils.azure_utils as azure_utils


# ======================================================
# Fixtures
# ======================================================


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


# ======================================================
# get_azure_engine
# ======================================================


def test_get_azure_engine_creates_engine(fake_config, monkeypatch):
    fake_engine = MagicMock(spec=Engine)

    monkeypatch.setattr(
        azure_utils,
        "create_engine",
        lambda url, **kwargs: fake_engine,
    )

    engine = azure_utils.get_azure_engine(
        configs_path=str(fake_config),
        driver="Fake Driver",
    )

    assert engine is fake_engine


def test_get_azure_engine_missing_azure_key(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(KeyError):
        azure_utils.get_azure_engine(str(path))


def test_get_azure_engine_missing_required_fields(tmp_path):
    cfg = {"azure": {"server": "x"}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(ValueError):
        azure_utils.get_azure_engine(str(path))


# ======================================================
# _quote_ident
# ======================================================


def test_quote_ident_basic():
    assert azure_utils._quote_ident("col") == "[col]"


def test_quote_ident_escapes_bracket():
    assert azure_utils._quote_ident("weird]name") == "[weird]]name]"


def test_quote_ident_none_raises():
    with pytest.raises(ValueError):
        azure_utils._quote_ident(None)


# ======================================================
# write_sql_table
# ======================================================


def test_write_sql_table_noop_on_empty_df(mock_engine):
    df = pd.DataFrame()

    with patch("pandas.DataFrame.to_sql") as mock_to_sql:
        azure_utils.write_sql_table(mock_engine, "tbl", df)
        mock_to_sql.assert_not_called()


def test_write_sql_table_overwrite(mock_engine):
    df = pd.DataFrame({"a": [1, 2]})

    with patch("pandas.DataFrame.to_sql") as mock_to_sql:
        azure_utils.write_sql_table(
            engine=mock_engine,
            table_name="tbl",
            df=df,
            overwrite=True,
        )

        mock_to_sql.assert_called_once()
        _, kwargs = mock_to_sql.call_args

        assert kwargs["if_exists"] == "replace"
        assert kwargs["index"] is False


from unittest.mock import patch


def test_write_sql_table_append(mock_engine):
    df = pd.DataFrame({"a": [1, 2]})

    with (
        patch("utils.azure_utils.inspect") as mock_inspect,
        patch("pandas.DataFrame.to_sql") as mock_to_sql,
    ):

        mock_inspector = mock_inspect.return_value
        mock_inspector.get_columns.return_value = [{"name": "a"}]

        azure_utils.write_sql_table(
            engine=mock_engine,
            table_name="tbl",
            df=df,
            overwrite=False,
        )

        mock_to_sql.assert_called_once()
        _, kwargs = mock_to_sql.call_args

        assert kwargs["if_exists"] == "append"
        assert kwargs["index"] is False


def test_write_sql_table_creates_index_when_specified(mock_engine):
    df = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-01"]})

    with (
        patch("pandas.DataFrame.to_sql") as mock_to_sql,
        patch("utils.azure_utils.ensure_index") as mock_ensure_index,
    ):
        azure_utils.write_sql_table(
            engine=mock_engine,
            table_name="prices",
            df=df,
            overwrite=True,
            index_spec={
                "name": "IX_prices_ticker_date",
                "columns": ["TICKER", "DATE"],
            },
        )

        mock_to_sql.assert_called_once()
        mock_ensure_index.assert_called_once_with(
            engine=mock_engine,
            table_name="prices",
            schema="dbo",
            index_name="IX_prices_ticker_date",
            columns=["TICKER", "DATE"],
        )


# ======================================================
# read_sql_table
# ======================================================


def test_read_sql_table_by_name(mock_engine):
    fake_df = pd.DataFrame({"x": [1, 2]})

    with patch("pandas.read_sql_query", return_value=fake_df) as mock_read:
        out = azure_utils.read_sql_table(
            engine=mock_engine,
            table_name="tbl",
        )

        mock_read.assert_called_once()
        assert out.equals(fake_df)


def test_read_sql_table_by_query(mock_engine):
    fake_df = pd.DataFrame({"x": [1]})

    with patch("pandas.read_sql_query", return_value=fake_df):
        out = azure_utils.read_sql_table(
            engine=mock_engine,
            query="SELECT 1 AS x",
        )

        assert out.iloc[0]["x"] == 1


def test_read_sql_table_requires_input(mock_engine):
    with pytest.raises(ValueError):
        azure_utils.read_sql_table(mock_engine)


# ======================================================
# delete_sql_rows
# ======================================================


def test_delete_sql_rows_executes(mock_engine):
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    azure_utils.delete_sql_rows(
        engine=mock_engine,
        table_name="tbl",
        where_clause="x > 1",
    )

    mock_conn.exec_driver_sql.assert_called_once()
    sql = mock_conn.exec_driver_sql.call_args[0][0]
    assert "DELETE FROM" in sql
    assert "WHERE x > 1" in sql


def test_delete_sql_rows_requires_where_clause(mock_engine):
    with pytest.raises(ValueError):
        azure_utils.delete_sql_rows(
            engine=mock_engine,
            table_name="tbl",
            where_clause=" ",
        )


# ======================================================
# execute_sql
# ======================================================


def test_execute_sql_executes_without_params(mock_engine):
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    azure_utils.execute_sql(
        engine=mock_engine,
        sql="CREATE TABLE x (y INT)",
    )

    mock_conn.exec_driver_sql.assert_called_once()


def test_execute_sql_executes_with_params(mock_engine):
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    azure_utils.execute_sql(
        engine=mock_engine,
        sql="SELECT :x",
        params={"x": 1},
    )

    mock_conn.execute.assert_called_once()
