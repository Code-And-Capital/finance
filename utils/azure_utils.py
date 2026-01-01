from typing import Optional, Dict

import pandas as pd
import os
from sqlalchemy import create_engine, text
from sqlalchemy import types as satypes
from sqlalchemy.engine import Engine, URL
from sqlalchemy.sql.type_api import TypeEngine

import utils.configs_reader as configs_reader

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

    Parameters
    ----------
    configs_path : str
        Path to the JSON configuration file.
    driver : str, default "ODBC Driver 18 for SQL Server"
        ODBC driver installed on the system.
    encrypt : bool, default True
        Whether to require encrypted connections (recommended for Azure SQL).
    trust_server_certificate : bool, default False
        Whether to trust the server certificate.
    connection_timeout : int, default 30
        Connection timeout in seconds.
    fast_executemany : bool, default True
        Enable pyodbc fast_executemany for bulk inserts.

    Returns
    -------
    sqlalchemy.engine.Engine
        SQLAlchemy engine connected to Azure SQL.
    """
    if not configs_path:
        configs_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..",
                "configs",
                "configs.json",
            )
        )
    configs = configs_reader.read_json_configs(configs_path)

    try:
        azure_cfg = configs["azure"]
    except KeyError as e:
        raise KeyError('Config missing required top-level key "azure".') from e

    server = azure_cfg.get("server")
    database = azure_cfg.get("database")
    username = azure_cfg.get("username")
    password = azure_cfg.get("password")

    missing = [
        k
        for k, v in {
            "azure.server": server,
            "azure.database": database,
            "azure.username": username,
            "azure.password": password,
        }.items()
        if not v
    ]

    if missing:
        raise ValueError(
            f"Missing Azure SQL connection settings in config: {', '.join(missing)}"
        )

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
# Helpers
# -----------------------------


def _quote_ident(name: str) -> str:
    """
    Safely quote a SQL Server identifier (table, schema, or column name).

    SQL identifiers cannot be parameterized, so when they are constructed
    dynamically they must be quoted correctly to avoid syntax errors.

    This function wraps the identifier in SQL Server brackets and escapes
    any closing bracket characters.

    Parameters
    ----------
    name : str
        Identifier to quote (e.g., schema, table, column).

    Returns
    -------
    str
        Quoted identifier (e.g., "[MyColumn]").

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


def ensure_index(
    engine: Engine,
    table_name: str,
    index_name: str,
    columns: list[str],
    schema: str = "dbo",
) -> None:
    """
    Create an index if it doesn't already exist (safe to rerun)

    Use after creating/replacing a table (e.g., when overwrite=True). SQL Server
    will automatically maintain the index on future inserts/updates/deletes.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine connected to Azure SQL.
    table_name : str
        Target table name (without schema).
    index_name : str
        Name of the index to create.
    columns : list[str]
        Column names to use as the index key, in order (e.g. ["TICKER", "DATE"]).
    schema : str, default "dbo"
        Schema name.
    """
    cols_sql = ", ".join(_quote_ident(c) for c in columns)
    index_query = f"""
    IF NOT EXISTS (
        SELECT 1
        FROM sys.indexes
        WHERE name = '{index_name}'
          AND object_id = OBJECT_ID('{schema}.{table_name}')
    )
    BEGIN
        CREATE INDEX {index_name}
        ON {_quote_ident(schema)}.{_quote_ident(table_name)} ({cols_sql});
    END
    """
    execute_sql(engine, index_query)


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
    dtype_overrides: Optional[Dict[str, TypeEngine]] = None,
    index_spec: Optional[dict] = None,
) -> None:
    """
    Write a pandas DataFrame to an Azure SQL / SQL Server table.

    Behavior
    --------
    - overwrite=False (default): Append rows to an existing table.
    - overwrite=True: Drop and recreate the table, then insert rows.
        * Allows explicit column types via `dtype_overrides` during table creation.
        * Optionally creates indexes via `index_spec` after table creation.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine connected to Azure SQL / SQL Server.
    table_name : str
        Target table name (without schema).
    df : pandas.DataFrame
        Data to write.
    schema : str, default "dbo"
        SQL schema.
    overwrite : bool, default False
        If True, drop and recreate the table (replace). If False, append.
    chunksize : int or None, default 1000
        Batch size for inserts.
    dtype_overrides : dict or None, default None
        Optional mapping of column name -> SQLAlchemy type, used only when
        overwrite=True (table creation). Example:
            {"TICKER": satypes.VARCHAR(16),
             "DATE": satypes.Date()}
        Any keys not present in `df.columns` are ignored.
    index_spec : dict or None, default None
        Optional index specification to create after overwrite=True.
        Format: {"name": "<index_name>", "columns": ["col1", "col2", ...]}.

    Notes
    -----
    - Indexes are maintained automatically by SQL Server after creation.
    - If df is empty, no action is taken.
    """
    if df is None or df.empty:
        return

    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    dtype_map = None
    if dtype_overrides:
        dtype_map = {k: v for k, v in dtype_overrides.items() if k in df.columns}

    # Create or replace table
    if overwrite:
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists="replace",
            index=False,
            chunksize=chunksize,
            dtype=dtype_map,
        )

        # Ensure specified index exists, otherwise, create it
        if index_spec:
            ensure_index(
                engine=engine,
                table_name=table_name,
                schema=schema,
                index_name=index_spec["name"],
                columns=index_spec["columns"],
            )
        return

    # Append to existing table
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

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine.
    table_name : str, optional
        Table name to read from. If provided, `query` is ignored.
    query : str, optional
        SQL query to execute.
    schema : str, default "dbo"
        Schema name.
    coerce_numeric : bool, default True
        Attempt to coerce columns to numeric types where possible.

    Returns
    -------
    pandas.DataFrame
        Query results.
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

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine connected to Azure SQL / SQL Server.
    table_name : str
        Target table name (without schema).
    where_clause : str
        SQL WHERE clause (without the "WHERE" keyword), e.g.
        "asof_date < '2020-01-01'" or "ticker = 'AAPL'".
    schema : str, default "dbo"
        Schema name.
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

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine.
    sql : str
        SQL statement to execute.
    params : dict, optional
        Optional bound parameters for the SQL statement.
    """
    with engine.begin() as conn:
        if params:
            conn.execute(text(sql), params)
        else:
            conn.exec_driver_sql(sql)
