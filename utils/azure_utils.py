import os
from typing import Optional, Dict

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

import finance.utils.configs_reader as configs_reader

# -----------------------------
# Connection / Engine helpers
# -----------------------------


def get_azure_engine(
    configs_path: str,
    driver: str = "ODBC Driver 18 for SQL Server",
    encrypt: bool = True,
    trust_server_certificate: bool = False,
    connection_timeout: int = 30,
    fast_executemany: bool = True,
) -> Engine:
    """
    Create a SQLAlchemy Engine for Azure SQL Database via pyodbc.

    Args:
        configs_path: Path to the JSON config file
        driver: ODBC driver name installed on your machine
        encrypt: Should be True for Azure SQL
        trust_server_certificate: Normally False
        connection_timeout: ODBC connection timeout seconds
        fast_executemany: Improves bulk insert performance

    Returns:
        SQLAlchemy Engine
    """

    configs = configs_reader.read_json_configs(configs_path)

    try:
            azure_cfg = configs["azure"]
    except KeyError as e:
        raise KeyError('Config missing required top-level key "azure".') from e

    server = azure_cfg.get("server")
    database = azure_cfg.get("database")
    username = azure_cfg.get("username")
    password = azure_cfg.get("password")

    missing = [k for k, v in {
        "azure.server": server,
        "azure.database": database,
        "azure.username": username,
        "azure.password": password,
    }.items() if not v]

    if missing:
        raise ValueError(f"Missing Azure SQL connection settings in config: {', '.join(missing)}")

    conn_str = (
        f"Driver={{{driver}}};"
        f"Server=tcp:{server},1433;"
        f"Database={database};"
        f"Uid={username};"
        f"Pwd={password};"
        f"Encrypt={'yes' if encrypt else 'no'};"
        f"TrustServerCertificate={'yes' if trust_server_certificate else 'no'};"
        f"Connection Timeout={int(connection_timeout)};"
    )

    url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
    engine = create_engine(url, fast_executemany=fast_executemany)
    return engine


# -----------------------------
# Type inference for new columns
# -----------------------------


def _infer_sqlserver_type(series: pd.Series) -> str:
    """
    Infer a reasonable SQL Server type for a pandas Series. Used only when
    adding new columns

    Biased toward types that are safe and broadly compatible.
    """
    dtype = series.dtype

    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    if pd.api.types.is_bool_dtype(dtype):
        return "BIT"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME2"
    if pd.api.types.is_timedelta64_dtype(dtype):
        return "BIGINT"

    # object/string/category fallback
    return "NVARCHAR(MAX)"


def _quote_ident(name: str) -> str:
    """
    Safely quote a SQL Server identifier (table, schema, or column name).

    SQL identifiers cannot be parameterized, so when they are constructed
    dynamically they must be quoted correctly to avoid syntax errors.

    This function wraps the identifier in SQL Server brackets and escapes
    any closing bracket characters.

    Examples
    --------
    >>> _quote_ident("prices_daily")
    "[prices_daily]"

    >>> _quote_ident("order")          # reserved keyword
    "[order]"

    >>> _quote_ident("City Population")  # spaces in name
    "[City Population]"

    >>> _quote_ident("weird]name")     # internal bracket is escaped
    "[weird]]name]"

    """
    if name is None:
        raise ValueError("Identifier cannot be None")
    return "[" + name.replace("]", "]]") + "]"


# -----------------------------
# Core utilities
# -----------------------------


def write_sql_table(
    engine: Engine,
    table_name: str,
    df: pd.DataFrame,
    schema: str = "dbo",
    overwrite: bool = False,
    chunksize: Optional[int] = 1000,
) -> None:
    """
    Write a pandas DataFrame to an Azure SQL table.

    If `overwrite=True`, the table is dropped and recreated.
    Otherwise, the table is created if missing, new columns are added
    as needed, and rows are appended.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine connected to Azure SQL.
    table_name : str
        Target table name (without schema).
    df : pandas.DataFrame
        Data to write.
    schema : str, default "dbo"
        SQL schema.
    overwrite : bool, default False
        Drop and recreate the table before writing.
    chunksize : int or None, default 1000
        Batch size for inserts.

    Notes
    -----
    - New columns are added via `ALTER TABLE` with inferred SQL Server types.
    - Operations that modify schema are executed inside a transaction.
    - No action is taken if the DataFrame is empty.
    """
    if df is None or df.empty:
        return

    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    if overwrite:
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists="replace",
            index=False,
            chunksize=chunksize,
        )
        return

    # Check table exists + add missing columns
    with engine.begin() as conn:
        table_exists = (
            conn.execute(
                text(
                    """
                SELECT 1
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
            """
                ),
                {"schema": schema, "table": table_name},
            ).scalar()
            is not None
        )

        if table_exists:
            existing_cols = {
                r[0]
                for r in conn.execute(
                    text(
                        """
                        SELECT COLUMN_NAME
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
                    """
                    ),
                    {"schema": schema, "table": table_name},
                ).fetchall()
            }

            missing_cols = [c for c in df.columns if c not in existing_cols]
            for col in missing_cols:
                sql_type = _infer_sqlserver_type(df[col])
                conn.exec_driver_sql(
                    f"ALTER TABLE {_quote_ident(schema)}.{_quote_ident(table_name)} "
                    f"ADD {_quote_ident(col)} {sql_type} NULL;"
                )

    # Write rows
    if not table_exists:
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists="replace",
            index=False,
            chunksize=chunksize,
        )
    else:
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists="append",
            index=False,
            chunksize=chunksize,
        )


def read_sql_table(
    engine: Engine,
    table_name: Optional[str] = None,
    query: Optional[str] = None,
    schema: str = "dbo",
    coerce_numeric: bool = True,
) -> pd.DataFrame:
    """
    Read data from Azure SQL into a pandas DataFrame.

    Provide either:
      - table_name -> SELECT * FROM schema.table_name
      - query -> any SQL query

    Args:
        engine: SQLAlchemy Engine
        table_name: Table name (optional)
        query: SQL query (optional; ignored if table_name provided)
        schema: Schema name
        coerce_numeric: Attempt to convert columns to numeric where possible (sqlite-like behavior)

    Returns:
        DataFrame
    """
    if table_name:
        query = f"SELECT * FROM {_quote_ident(schema)}.{_quote_ident(table_name)}"

    if not query:
        raise ValueError("You must provide either table_name or query.")

    df = pd.read_sql_query(sql=text(query), con=engine)

    if coerce_numeric:
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    return df


def delete_sql_rows(
    engine: Engine,
    table_name: str,
    where_clause: str,
    schema: str = "dbo",
) -> None:
    """
    Delete rows from an Azure SQL table based on a WHERE clause.

    Args:
        engine: SQLAlchemy Engine
        table_name: Table name
        where_clause: e.g. "asof_date < '2020-01-01'" or "ticker = 'AAPL'"
        schema: Schema name
    """
    if not where_clause or not where_clause.strip():
        raise ValueError(
            "where_clause must be a non-empty string (refusing to run DELETE without a filter)."
        )

    sql = f"DELETE FROM {_quote_ident(schema)}.{_quote_ident(table_name)} WHERE {where_clause};"
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def execute_sql(
    engine: Engine,
    sql: str,
    params: Optional[Dict] = None,
) -> None:
    """
    Execute an arbitrary SQL statement inside a transaction.

    Useful for one-off DDL (CREATE TABLE, CREATE INDEX, etc.).
    """
    with engine.begin() as conn:
        if params:
            conn.execute(text(sql), params)
        else:
            conn.exec_driver_sql(sql)
