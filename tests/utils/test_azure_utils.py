import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import types as satypes
from sqlalchemy.engine import Engine

from connectors.azure_data_source import AzureDataSource


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


@pytest.fixture
def source():
    return AzureDataSource()


def test_get_engine_creates_engine(fake_config, source, monkeypatch):
    fake_engine = MagicMock(spec=Engine)

    monkeypatch.setattr(
        "connectors.azure_data_source.create_engine",
        lambda url, **kwargs: fake_engine,
    )

    engine = source.get_engine(configs_path=str(fake_config), driver="Fake Driver")
    assert engine is fake_engine
    assert source.engine is fake_engine


def test_connect_sets_engine(fake_config, source, monkeypatch):
    fake_engine = MagicMock(spec=Engine)
    monkeypatch.setattr(
        source,
        "get_engine",
        lambda **kwargs: fake_engine,
    )

    out = source.connect(configs_path=str(fake_config))
    assert out is source
    assert source.engine is fake_engine


def test_get_engine_missing_azure_key(tmp_path, source):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(KeyError):
        source.get_engine(str(path))


def test_get_engine_missing_required_fields(tmp_path, source):
    cfg = {"azure": {"server": "x"}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(ValueError):
        source.get_engine(str(path))


def test_write_sql_table_noop_on_empty_df(mock_engine, source):
    source.set_engine(mock_engine)
    df = pd.DataFrame()

    with patch("pandas.DataFrame.to_sql") as mock_to_sql:
        source.write_sql_table(table_name="tbl", df=df)
        mock_to_sql.assert_not_called()


def test_write_sql_table_overwrite(mock_engine, source):
    source.set_engine(mock_engine)
    df = pd.DataFrame({"a": [1, 2]})

    with (
        patch("pandas.DataFrame.to_sql") as mock_to_sql,
        patch.object(source, "_validate_write_sql_table") as mock_validate,
    ):
        source.write_sql_table(table_name="tbl", df=df, overwrite=True)

        mock_to_sql.assert_called_once()
        mock_validate.assert_called_once()
        _, kwargs = mock_to_sql.call_args

        assert kwargs["if_exists"] == "replace"
        assert kwargs["index"] is False


def test_write_sql_table_append(mock_engine, source):
    source.set_engine(mock_engine)
    df = pd.DataFrame({"a": [1, 2]})

    with (
        patch("connectors.azure_data_source.inspect") as mock_inspect,
        patch("pandas.DataFrame.to_sql") as mock_to_sql,
        patch.object(source, "_validate_write_sql_table") as mock_validate,
    ):

        mock_inspector = mock_inspect.return_value
        mock_inspector.get_columns.return_value = [{"name": "a"}]

        source.write_sql_table(table_name="tbl", df=df, overwrite=False)

        mock_to_sql.assert_called_once()
        mock_validate.assert_called_once()
        _, kwargs = mock_to_sql.call_args

        assert kwargs["if_exists"] == "append"
        assert kwargs["index"] is False


def test_write_sql_table_append_uses_dtype_overrides(mock_engine, source):
    source.set_engine(mock_engine)
    df = pd.DataFrame({"DATE": [pd.Timestamp("2026-03-01")]})

    with (
        patch("connectors.azure_data_source.inspect") as mock_inspect,
        patch("pandas.DataFrame.to_sql") as mock_to_sql,
        patch.object(source, "_validate_write_sql_table") as mock_validate,
    ):
        mock_inspector = mock_inspect.return_value
        mock_inspector.get_columns.return_value = [{"name": "DATE"}]

        source.write_sql_table(
            table_name="tbl",
            df=df,
            overwrite=False,
            dtype_overrides={"DATE": satypes.Date()},
        )

        mock_to_sql.assert_called_once()
        mock_validate.assert_called_once()
        _, kwargs = mock_to_sql.call_args
        assert "dtype" in kwargs
        assert "DATE" in kwargs["dtype"]


def test_write_sql_table_coerces_sql_date_columns_to_date_values(mock_engine, source):
    source.set_engine(mock_engine)
    df = pd.DataFrame({"DATE": [pd.Timestamp("2026-03-01 08:23:04")]})

    with (
        patch("connectors.azure_data_source.inspect") as mock_inspect,
        patch("pandas.DataFrame.to_sql") as mock_to_sql,
        patch.object(source, "_validate_write_sql_table") as mock_validate,
    ):
        mock_inspector = mock_inspect.return_value
        mock_inspector.get_columns.return_value = [
            {"name": "DATE", "type": satypes.Date()}
        ]

        source.write_sql_table(table_name="tbl", df=df, overwrite=False)

        mock_to_sql.assert_called_once()
        mock_validate.assert_called_once()
        written_df = mock_validate.call_args.kwargs["df"]
        assert str(written_df.loc[0, "DATE"]) == "2026-03-01"


def test_write_sql_table_executes_index_query(mock_engine, source):
    source.set_engine(mock_engine)
    df = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-01"]})

    with (
        patch("pandas.DataFrame.to_sql") as mock_to_sql,
        patch.object(source, "execute_sql") as mock_execute_sql,
        patch.object(source, "_validate_write_sql_table") as mock_validate,
    ):
        source.write_sql_table(
            table_name="prices",
            df=df,
            overwrite=True,
            index_query="CREATE INDEX ...",
        )

        mock_to_sql.assert_called_once()
        mock_execute_sql.assert_called_once_with(
            sql="CREATE INDEX ...", engine=mock_engine
        )
        mock_validate.assert_called_once()


def test_validate_write_sql_table_passes_when_row_count_matches(mock_engine, source):
    source.set_engine(mock_engine)
    df = pd.DataFrame({"TICKER": ["AAPL"], "DATE": ["2024-01-01"], "X": [1]})

    with (
        patch.object(source, "read_sql_table", return_value=df.copy()) as mock_read,
        patch("connectors.azure_data_source.log") as mock_log,
    ):
        source._validate_write_sql_table(
            schema="dbo",
            table_name="prices",
            df=df,
            engine=mock_engine,
        )

    mock_read.assert_called_once()
    assert any(
        "Write validation passed for dbo.prices" in call.args[0]
        for call in mock_log.call_args_list
    )


def test_validate_write_sql_table_skips_when_required_columns_missing(
    mock_engine, source
):
    source.set_engine(mock_engine)
    df = pd.DataFrame({"A": [1]})

    with (
        patch.object(source, "read_sql_table") as mock_read,
        patch("connectors.azure_data_source.log") as mock_log,
    ):
        source._validate_write_sql_table(
            schema="dbo",
            table_name="address",
            df=df,
            engine=mock_engine,
        )

    mock_read.assert_not_called()
    assert any(
        "Write validation skipped for dbo.address" in call.args[0]
        for call in mock_log.call_args_list
    )


def test_read_sql_table_by_query(mock_engine, source):
    source.set_engine(mock_engine)
    fake_df = pd.DataFrame({"x": [1]})

    with patch("pandas.read_sql_query", return_value=fake_df):
        out = source.read_sql_table(query="SELECT 1 AS x")
        assert out.iloc[0]["x"] == 1
        assert source.last_query == "SELECT 1 AS x"


def test_read_sql_table_requires_query(mock_engine, source):
    source.set_engine(mock_engine)
    with pytest.raises(ValueError, match="query must be provided"):
        source.read_sql_table(query="")


def test_read_sql_table_requires_engine(source):
    with pytest.raises(ValueError, match="No Azure engine is set"):
        source.read_sql_table(query="SELECT 1")


def test_delete_sql_rows_executes_query(mock_engine, source):
    source.set_engine(mock_engine)
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    source.delete_sql_rows(query="DELETE FROM [dbo].[tbl] WHERE x > 1;")

    mock_conn.exec_driver_sql.assert_called_once()
    assert source.last_query == "DELETE FROM [dbo].[tbl] WHERE x > 1;"


def test_delete_sql_rows_requires_query(mock_engine, source):
    source.set_engine(mock_engine)
    with pytest.raises(ValueError, match="query must be a non-empty SQL statement"):
        source.delete_sql_rows(query=" ")


def test_execute_sql_executes_without_params(mock_engine, source):
    source.set_engine(mock_engine)
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    source.execute_sql(sql="CREATE TABLE x (y INT)")

    mock_conn.exec_driver_sql.assert_called_once()


def test_execute_sql_executes_with_params(mock_engine, source):
    source.set_engine(mock_engine)
    mock_conn = MagicMock()
    mock_engine.begin.return_value.__enter__.return_value = mock_conn

    source.execute_sql(sql="SELECT :x", params={"x": 1})

    mock_conn.execute.assert_called_once()


def test_disconnect_disposes_engine(mock_engine, source):
    source.set_engine(mock_engine)
    source.disconnect()
    mock_engine.dispose.assert_called_once()
    assert source.engine is None
