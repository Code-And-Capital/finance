"""Stateful Azure SQL data source implementation."""

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy import types as satypes
from sqlalchemy.engine import Engine, URL
from sqlalchemy.sql.type_api import TypeEngine

import utils.dataframe_utils as dataframe_utils
from config.configs import Configs
from sql.script_factory import SQLClient
from utils.logging import log


class AzureDataSource:
    """Stateful adapter for Azure SQL connection and execution.

    Instances keep the active SQLAlchemy engine on `self.engine` and track the
    most recent executed SQL text on `self.last_query`.
    """

    def __init__(self, engine: Engine | None = None) -> None:
        """Initialize the data source.

        Parameters
        ----------
        engine
            Optional pre-constructed SQLAlchemy engine.
        """
        self.engine: Engine | None = engine
        self.last_query: str | None = None
        self.sql_client = SQLClient()

    @staticmethod
    def _default_configs_path() -> str:
        """Return the default repository config path for Azure credentials."""
        return str(
            (
                Path(__file__).resolve().parent.parent / "config" / "configs.json"
            ).resolve()
        )

    def set_engine(self, engine: Engine) -> AzureDataSource:
        """Assign an existing engine to the instance and return `self`."""
        self.engine = engine
        return self

    def connect(
        self,
        configs_path: str | None = None,
        driver: str = "ODBC Driver 18 for SQL Server",
        encrypt: bool = True,
        trust_server_certificate: bool = False,
        connection_timeout: int = 30,
        fast_executemany: bool = True,
    ) -> AzureDataSource:
        """Create and store an engine on the instance.

        Returns
        -------
        AzureDataSource
            Current instance for fluent chaining.
        """
        self.engine = self.get_engine(
            configs_path=configs_path,
            driver=driver,
            encrypt=encrypt,
            trust_server_certificate=trust_server_certificate,
            connection_timeout=connection_timeout,
            fast_executemany=fast_executemany,
        )
        return self

    def disconnect(self) -> AzureDataSource:
        """Dispose and clear the currently stored engine if present."""
        if self.engine is not None:
            self.engine.dispose()
        self.engine = None
        return self

    def get_engine(
        self,
        configs_path: str | None = None,
        driver: str = "ODBC Driver 18 for SQL Server",
        encrypt: bool = True,
        trust_server_certificate: bool = False,
        connection_timeout: int = 30,
        fast_executemany: bool = True,
    ) -> Engine:
        """Build a SQLAlchemy engine and store it on `self.engine`."""
        config_path = configs_path or self._default_configs_path()
        configs = Configs(path=config_path).load().as_dict()

        try:
            azure_cfg = configs["azure"]
        except KeyError as exc:
            raise KeyError('Config missing required top-level key "azure".') from exc

        server = azure_cfg.get("server")
        database = azure_cfg.get("database")
        username = azure_cfg.get("username")
        password = azure_cfg.get("password")

        missing = [
            key
            for key, value in {
                "azure.server": server,
                "azure.database": database,
                "azure.username": username,
                "azure.password": password,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing Azure SQL connection settings in config: {', '.join(missing)}"
            )

        connection_string = (
            f"Driver={{{driver}}};"
            f"Server=tcp:{server},1433;"
            f"Database={database};"
            f"Uid={username};"
            f"Pwd={password};"
            f"Encrypt={'yes' if encrypt else 'no'};"
            f"TrustServerCertificate={'yes' if trust_server_certificate else 'no'};"
            f"Connection Timeout={int(connection_timeout)};"
        )
        url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
        self.engine = create_engine(url, fast_executemany=fast_executemany)
        return self.engine

    def _resolve_engine(self, engine: Engine | None) -> Engine:
        """Resolve the engine parameter or fallback to `self.engine`."""
        resolved = engine or self.engine
        if resolved is None:
            raise ValueError(
                "No Azure engine is set. Call connect() or pass engine explicitly."
            )
        return resolved

    @staticmethod
    def _normalize_sql_nulls(df: pd.DataFrame) -> pd.DataFrame:
        """Convert pandas missing sentinels to Python ``None`` for DB-API bindings."""
        out = df.copy()
        for column in out.columns:
            if isinstance(out[column].dtype, pd.DatetimeTZDtype):
                out[column] = out[column].dt.tz_convert(None)
            # Guard against non-finite numerics causing DBAPI overflow errors.
            # pyodbc cannot bind inf/-inf into numeric SQL columns.
            numeric = pd.to_numeric(out[column], errors="coerce")
            if numeric.notna().any():
                inf_mask = numeric.isin([float("inf"), float("-inf")])
                if inf_mask.any():
                    out.loc[inf_mask, column] = None
        return out.where(pd.notna(out), None)

    @staticmethod
    def _coerce_temporal_columns_to_sql_types(
        df: pd.DataFrame,
        *,
        column_types: dict[str, TypeEngine] | None = None,
    ) -> pd.DataFrame:
        """Coerce temporal columns to match SQLAlchemy column types.

        DATE columns are converted to Python ``date`` objects (no time component).
        DATETIME columns are converted to timezone-naive pandas datetimes.
        """
        if not column_types:
            return df

        out = df.copy()
        for column, sql_type in column_types.items():
            if column not in out.columns:
                continue

            if isinstance(sql_type, satypes.Date) and not isinstance(
                sql_type, satypes.DateTime
            ):
                parsed = dataframe_utils.coerce_datetime_series(
                    out[column], errors="coerce"
                )
                out[column] = parsed.dt.date
            elif isinstance(sql_type, satypes.DateTime):
                out[column] = dataframe_utils.coerce_datetime_series(
                    out[column], errors="coerce"
                )

        return out

    def _validate_write_sql_table(
        self,
        *,
        schema: str,
        table_name: str,
        df: pd.DataFrame,
        engine: Engine,
        ticker_column: str = "TICKER",
        date_column: str = "DATE",
    ) -> None:
        """Validate that a write is queryable in Azure SQL by ticker/date filters."""
        if df.empty:
            log(
                f"Write validation skipped for {schema}.{table_name}: dataframe is empty."
            )
            return

        if ticker_column not in df.columns or date_column not in df.columns:
            log(
                f"Write validation skipped for {schema}.{table_name}: "
                f"missing required columns '{ticker_column}' and/or '{date_column}'.",
                type="warning",
            )
            return

        ticker_values = (
            df[ticker_column]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        date_values = (
            dataframe_utils.coerce_datetime_series(df[date_column], errors="coerce")
            .dropna()
            .dt.date.astype(str)
            .unique()
            .tolist()
        )

        if not ticker_values or not date_values:
            log(
                f"Write validation skipped for {schema}.{table_name}: "
                "no valid ticker/date keys found in dataframe.",
                type="warning",
            )
            return

        query = self.sql_client.build_validate_ticker_date_query(
            table_name=table_name,
            schema=schema,
            ticker_column=ticker_column,
            date_column=date_column,
            ticker_values=ticker_values,
            date_values=date_values,
        )

        written_df = self.read_sql_table(
            query=query, coerce_numeric=False, engine=engine
        )
        if (
            ticker_column not in written_df.columns
            or date_column not in written_df.columns
        ):
            log(
                f"Write validation skipped for {schema}.{table_name}: "
                f"database result missing required columns '{ticker_column}' and/or '{date_column}'.",
                type="warning",
            )
            return

        payload_keys = pd.DataFrame(
            {
                ticker_column: df[ticker_column].astype(str).str.strip(),
                date_column: pd.to_datetime(
                    df[date_column], errors="coerce"
                ).dt.date.astype("string"),
            }
        ).replace({"": pd.NA})
        payload_keys = payload_keys.dropna(subset=[ticker_column, date_column])
        payload_pairs = set(payload_keys.itertuples(index=False, name=None))

        database_keys = pd.DataFrame(
            {
                ticker_column: written_df[ticker_column].astype(str).str.strip(),
                date_column: pd.to_datetime(
                    written_df[date_column], errors="coerce"
                ).dt.date.astype("string"),
            }
        ).replace({"": pd.NA})
        database_keys = database_keys.dropna(subset=[ticker_column, date_column])
        database_pairs = set(database_keys.itertuples(index=False, name=None))

        missing_pairs = [pair for pair in payload_pairs if pair not in database_pairs]
        if not missing_pairs:
            log(
                f"Write validation passed for {schema}.{table_name}: "
                f"all {len(payload_pairs)} ticker-date pairs found in database slice.",
            )
        else:
            log(
                f"Write validation mismatch for {schema}.{table_name}: "
                f"{len(missing_pairs)}/{len(payload_pairs)} ticker-date pairs not found in database slice.",
                type="warning",
            )

    def write_sql_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        schema: str = "dbo",
        overwrite: bool = False,
        chunksize: int | None = 1000,
        dtype_overrides: dict[str, TypeEngine] | None = None,
        index_query: str | None = None,
        align_missing_to_table: bool = True,
        validation_ticker_column: str = "TICKER",
        validation_date_column: str = "DATE",
        engine: Engine | None = None,
    ) -> None:
        """Write a DataFrame into an Azure SQL table.

        Parameters
        ----------
        table_name
            Target table name.
        df
            DataFrame payload to persist.
        schema
            Target schema name.
        overwrite
            If True, replace table contents.
        chunksize
            Batch row count for pandas `to_sql`.
        dtype_overrides
            Optional SQLAlchemy dtype mapping.
        index_query
            Optional SQL statement executed after overwrite.
        align_missing_to_table
            Align append payload columns to existing table columns.
        validation_ticker_column
            Column used for post-write query validation ticker filtering.
        validation_date_column
            Column used for post-write query validation date filtering.
        engine
            Optional one-off engine override.
        """
        resolved_engine = self._resolve_engine(engine)

        if df is None or df.empty:
            return

        out = df.copy()
        out.columns = [str(column) for column in out.columns]
        table_column_types: dict[str, TypeEngine] = {}

        if align_missing_to_table and not overwrite:
            inspector = inspect(resolved_engine)
            try:
                inspected_columns = inspector.get_columns(table_name, schema=schema)
                table_columns = [column["name"] for column in inspected_columns]
                table_column_types = {
                    column["name"]: column["type"]
                    for column in inspected_columns
                    if "name" in column and "type" in column
                }
            except Exception:  # noqa: BLE001
                table_columns = []

            if table_columns:
                for column in table_columns:
                    if column not in out.columns:
                        out[column] = None
                out = out[[column for column in table_columns if column in out.columns]]

        dtype_map = None
        if dtype_overrides:
            dtype_map = {
                column: dtype
                for column, dtype in dtype_overrides.items()
                if column in out.columns
            }

        temporal_type_hints: dict[str, TypeEngine] = {}
        temporal_type_hints.update(table_column_types)
        if dtype_map:
            temporal_type_hints.update(dtype_map)
        out = self._coerce_temporal_columns_to_sql_types(
            out,
            column_types=temporal_type_hints,
        )
        out = self._normalize_sql_nulls(out)

        if overwrite:
            out.to_sql(
                table_name,
                resolved_engine,
                schema=schema,
                if_exists="replace",
                index=False,
                chunksize=chunksize,
                dtype=dtype_map,
            )
            if index_query:
                self.execute_sql(sql=index_query, engine=resolved_engine)
            self._validate_write_sql_table(
                schema=schema,
                table_name=table_name,
                df=out,
                engine=resolved_engine,
                ticker_column=validation_ticker_column,
                date_column=validation_date_column,
            )
            return

        out.to_sql(
            table_name,
            resolved_engine,
            schema=schema,
            if_exists="append",
            index=False,
            chunksize=chunksize,
            dtype=dtype_map,
        )
        self._validate_write_sql_table(
            schema=schema,
            table_name=table_name,
            df=out,
            engine=resolved_engine,
            ticker_column=validation_ticker_column,
            date_column=validation_date_column,
        )

    def read_sql_table(
        self,
        query: str,
        coerce_numeric: bool = True,
        engine: Engine | None = None,
    ) -> pd.DataFrame:
        """Execute a SELECT query and return results as a DataFrame."""
        if not query:
            raise ValueError("query must be provided")

        resolved_engine = self._resolve_engine(engine)
        self.last_query = query
        dataframe = pd.read_sql_query(sql=text(query), con=resolved_engine)

        if coerce_numeric:
            for column in dataframe.columns:
                try:
                    dataframe[column] = pd.to_numeric(dataframe[column])
                except (ValueError, TypeError):
                    pass

        return dataframe

    def delete_sql_rows(self, query: str, engine: Engine | None = None) -> None:
        """Execute a DELETE statement against Azure SQL."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty SQL statement")

        resolved_engine = self._resolve_engine(engine)
        self.last_query = query
        with resolved_engine.begin() as connection:
            connection.exec_driver_sql(query)

    def execute_sql(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        engine: Engine | None = None,
    ) -> None:
        """Execute arbitrary SQL (DDL/DML) in a transaction."""
        resolved_engine = self._resolve_engine(engine)
        self.last_query = sql
        with resolved_engine.begin() as connection:
            if params is not None:
                connection.execute(text(sql), params)
            else:
                connection.exec_driver_sql(sql)


default_azure_data_source = AzureDataSource()

__all__ = ["AzureDataSource", "default_azure_data_source", "satypes", "TypeEngine"]
