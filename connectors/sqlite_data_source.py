"""Stateful SQLite data source implementation."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing

import pandas as pd


class SQLiteDataSource:
    """Stateful adapter for SQLite table read/write/delete operations."""

    def __init__(self, database_name: str | None = None) -> None:
        """Initialize the SQLite data source.

        Parameters
        ----------
        database_name
            Optional logical database name or `.sqlite` path used as default.
        """
        self.database_name: str | None = database_name
        self.database_loc: str | None = (
            self.resolve_location(database_name) if database_name is not None else None
        )
        self.connection: sqlite3.Connection | None = None
        self.last_query: str | None = None

    @staticmethod
    def resolve_location(database_name: str) -> str:
        """Resolve a logical database name or explicit path to a `.sqlite` file."""
        if database_name.endswith(".sqlite") or os.path.sep in database_name:
            return database_name
        return os.path.join(
            os.path.abspath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
            ),
            "Data",
            f"{database_name}.sqlite",
        )

    def set_database(self, database_name: str) -> SQLiteDataSource:
        """Set the default database target and return `self` for chaining."""
        self.database_name = database_name
        self.database_loc = self.resolve_location(database_name)
        return self

    def connect(self, database_name: str | None = None) -> SQLiteDataSource:
        """Open and store a persistent SQLite connection on the instance."""
        if database_name is not None:
            self.set_database(database_name)
        if self.database_loc is None:
            raise ValueError(
                "No sqlite database is set. Call set_database() or pass database_name."
            )
        self.disconnect()
        self.connection = sqlite3.connect(self.database_loc)
        return self

    def disconnect(self) -> SQLiteDataSource:
        """Close and clear the stored SQLite connection, if present."""
        if self.connection is not None:
            self.connection.close()
        self.connection = None
        return self

    def _resolve_database_loc(self, database_name: str | None) -> str:
        """Resolve effective database location from parameter or instance state."""
        if database_name is not None:
            return self.resolve_location(database_name)
        if self.database_loc is None:
            raise ValueError(
                "No sqlite database is set. Call set_database() or pass database_name."
            )
        return self.database_loc

    def _read_sql(
        self, sql_query: str, database_name: str | None = None
    ) -> pd.DataFrame:
        """Execute a read query using either a persistent or transient connection."""
        self.last_query = sql_query
        if self.connection is not None and database_name is None:
            return pd.read_sql_query(sql_query, self.connection)

        database_loc = self._resolve_database_loc(database_name)
        with closing(sqlite3.connect(database=database_loc)) as connection:
            return pd.read_sql_query(sql_query, connection)

    def write_sql_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        overwrite: bool = False,
        database_name: str | None = None,
    ) -> None:
        """Write a DataFrame to a SQLite table, adding new columns when needed."""
        database_loc = self._resolve_database_loc(database_name)

        if self.connection is not None and database_name is None:
            connection = self.connection
            cursor = connection.cursor()

            if overwrite:
                df.to_sql(table_name, connection, if_exists="replace", index=False)
                return

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (table_name,),
            )
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                df.to_sql(table_name, connection, if_exists="replace", index=False)
                return

            cursor.execute(f"PRAGMA table_info({table_name});")
            existing_cols = [column[1] for column in cursor.fetchall()]
            new_cols = [column for column in df.columns if column not in existing_cols]
            for column in new_cols:
                cursor.execute(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{column}" TEXT;'
                )
            connection.commit()
            df.to_sql(table_name, connection, if_exists="append", index=False)
            return

        with closing(sqlite3.connect(database_loc)) as connection:
            cursor = connection.cursor()

            if overwrite:
                df.to_sql(table_name, connection, if_exists="replace", index=False)
                return

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (table_name,),
            )
            table_exists = cursor.fetchone() is not None
            if not table_exists:
                df.to_sql(table_name, connection, if_exists="replace", index=False)
                return

            cursor.execute(f"PRAGMA table_info({table_name});")
            existing_cols = [column[1] for column in cursor.fetchall()]
            new_cols = [column for column in df.columns if column not in existing_cols]
            for column in new_cols:
                cursor.execute(
                    f'ALTER TABLE "{table_name}" ADD COLUMN "{column}" TEXT;'
                )
            connection.commit()
            df.to_sql(table_name, connection, if_exists="append", index=False)

    def read_sql_table(
        self,
        table_name: str | None = None,
        query: str | None = None,
        database_name: str | None = None,
    ) -> pd.DataFrame:
        """Read a SQLite table or custom SQL query into a DataFrame."""
        sql_query = query
        if table_name:
            sql_query = f"SELECT * FROM {table_name}"
        if sql_query is None:
            raise ValueError("Either table_name or query must be provided.")

        dataframe = self._read_sql(sql_query, database_name=database_name)
        for column in dataframe.columns:
            try:
                dataframe[column] = dataframe[column].astype(float)
            except (ValueError, TypeError):
                pass

        return dataframe

    def delete_sql_rows(
        self,
        table_name: str | None = None,
        where_clause: str | None = None,
        database_name: str | None = None,
    ) -> None:
        """Delete rows from a table using a required WHERE clause."""
        if not table_name or not where_clause:
            raise ValueError("You must provide a table_name with a where_clause.")

        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        self.last_query = query

        if self.connection is not None and database_name is None:
            cursor = self.connection.cursor()
            cursor.execute(query)
            self.connection.commit()
            return

        database_loc = self._resolve_database_loc(database_name)
        with closing(sqlite3.connect(database_loc)) as connection:
            cursor = connection.cursor()
            cursor.execute(query)
            connection.commit()


default_sqlite_data_source = SQLiteDataSource()

__all__ = ["SQLiteDataSource", "default_sqlite_data_source"]
