"""Stateful SQL query construction and rendering utilities."""

import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

DateLike = str | date | datetime | pd.Timestamp
_PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class SQLClient:
    """Stateful SQL query builder and template renderer.

    This class tracks the current query in-memory (`self._query`) so callers can
    compose queries step-by-step and then access the final SQL via `query()`.
    """

    def __init__(
        self, sql_root: str | Path | None = None, query: str | None = None
    ) -> None:
        """Initialize the SQL client.

        Parameters
        ----------
        sql_root
            Optional root directory containing `.sql`/`.txt` query templates.
            Defaults to `<repo>/sql/templates`.
        query
            Optional initial SQL query to store as the current query.
        """
        default_root = Path(__file__).resolve().parent / "templates"
        self.sql_root = Path(sql_root) if sql_root is not None else default_root
        self._query = query

    def query(self) -> str:
        """Return the currently stored SQL query.

        Raises
        ------
        ValueError
            If no query has been set or rendered yet.
        """
        if not self._query:
            raise ValueError("No query is currently set on SQLClient")
        return self._query

    def clear(self) -> SQLClient:
        """Clear the currently stored query and return `self` for chaining."""
        self._query = None
        return self

    def set_query(self, query: str) -> SQLClient:
        """Set the current query string and return `self` for chaining."""
        self._query = query
        return self

    @staticmethod
    def quote_ident(name: str) -> str:
        """Quote a SQL Server identifier using square-bracket escaping."""
        if not isinstance(name, str) or not name:
            raise ValueError("Identifier must be a non-empty string")
        return "[" + name.replace("]", "]]") + "]"

    def resolve_sql_path(self, file_name: str) -> str:
        """Resolve an absolute path to a template file under `sql_root`."""
        return str((self.sql_root / file_name).resolve())

    def load_template(self, file_name: str) -> SQLClient:
        """Load a SQL template file into the current query state.

        Parameters
        ----------
        file_name
            Template file name relative to `sql_root`.

        Returns
        -------
        SQLClient
            The current instance for chaining.
        """
        path = Path(self.resolve_sql_path(file_name))
        if not path.exists():
            raise FileNotFoundError(f"Query file not found: {path}")
        self._query = path.read_text(encoding="utf-8")
        return self

    def _render(self, filters: dict[str, str]) -> SQLClient:
        """Render placeholders in the current query using provided filters.

        Parameters
        ----------
        filters
            Mapping of placeholder name to replacement SQL fragment.

        Returns
        -------
        SQLClient
            The current instance for chaining.
        """
        current_query = self.query()
        required_filters = set(_PLACEHOLDER_PATTERN.findall(current_query))
        missing = required_filters - filters.keys()
        if missing:
            raise ValueError(f"Missing required SQL filters: {sorted(missing)}")

        rendered = current_query
        for name in required_filters:
            rendered = rendered.replace(f"{{{name}}}", filters[name])

        self._query = rendered
        return self

    def render_sql_query(
        self, *, query_path: str | Path, filters: dict[str, str]
    ) -> str:
        """Render a SQL template from path and return the final query.

        This method updates the internal query state.
        """
        path = Path(query_path)
        if not path.exists():
            raise FileNotFoundError(f"Query file not found: {path}")

        self.set_query(path.read_text(encoding="utf-8"))
        self._render(filters)
        return self.query()

    @staticmethod
    def add_in_filter(column: str, values: Optional[Sequence[str]]) -> str:
        """Add an SQL `IN (...)` filter fragment for string values."""
        if not values:
            return ""
        escaped = [value.replace("'", "''") for value in values]
        joined = "', '".join(escaped)
        return f"AND {column} IN ('{joined}')"

    @staticmethod
    def add_date_filter(column: str, start_date: Optional[DateLike]) -> str:
        """Add an inclusive SQL date filter fragment."""
        if start_date is None:
            return ""

        try:
            timestamp = pd.Timestamp(start_date)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("start_date must be parseable as a date") from exc

        return f"AND {column} >= '{timestamp.date().isoformat()}'"

    @staticmethod
    def add_end_date_filter(column: str, end_date: Optional[DateLike]) -> str:
        """Add an inclusive SQL end-date filter fragment."""
        if end_date is None:
            return ""

        try:
            timestamp = pd.Timestamp(end_date)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("end_date must be parseable as a date") from exc

        return f"AND {column} <= '{timestamp.date().isoformat()}'"

    def build_select_all_query(self, *, table_name: str, schema: str = "dbo") -> str:
        """Build and store the standard `SELECT * FROM schema.table` query."""
        return self.render_sql_query(
            query_path=self.resolve_sql_path("base.txt"),
            filters={
                "schema": self.quote_ident(schema),
                "table_name": self.quote_ident(table_name),
            },
        )

    def build_select_with_filters_query(
        self,
        *,
        table_name: str,
        filters_sql: str = "",
        schema: str = "dbo",
    ) -> str:
        """Build and store `SELECT *` query with optional SQL filter fragments.

        Parameters
        ----------
        table_name
            Target table name.
        filters_sql
            SQL fragment(s) appended after ``WHERE 1 = 1``. Typical input is
            from ``add_in_filter`` and ``add_date_filter`` helpers.
        schema
            Target schema name.
        """
        return self.render_sql_query(
            query_path=self.resolve_sql_path("base_with_filters.txt"),
            filters={
                "schema": self.quote_ident(schema),
                "table_name": self.quote_ident(table_name),
                "filters_sql": filters_sql,
            },
        )

    def build_create_index_query(
        self,
        *,
        table_name: str,
        index_name: str,
        columns: list[str],
        schema: str = "dbo",
    ) -> str:
        """Build and store a `CREATE INDEX IF NOT EXISTS` style query template."""
        if not columns:
            raise ValueError("columns must be a non-empty list of column names")
        if not index_name:
            raise ValueError("index_name must be a non-empty string")

        columns_sql = ", ".join(self.quote_ident(column) for column in columns)
        return self.render_sql_query(
            query_path=self.resolve_sql_path("create_index.txt"),
            filters={
                "index_name": index_name,
                "schema": self.quote_ident(schema),
                "table_name": self.quote_ident(table_name),
                "cols_sql": columns_sql,
            },
        )

    def build_delete_query(
        self,
        *,
        table_name: str,
        where_clause: str,
        schema: str = "dbo",
    ) -> str:
        """Build and store a parameterized-safe DELETE statement skeleton."""
        if not table_name:
            raise ValueError("table_name must be provided")
        if not where_clause or not where_clause.strip():
            raise ValueError(
                "where_clause must be a non-empty string (refusing to run DELETE without a filter)."
            )

        self._query = (
            f"DELETE FROM {self.quote_ident(schema)}.{self.quote_ident(table_name)} "
            f"WHERE {where_clause};"
        )
        return self.query()

    def build_validate_ticker_date_query(
        self,
        *,
        table_name: str,
        ticker_values: Sequence[str],
        date_values: Sequence[str],
        schema: str = "dbo",
        ticker_column: str = "TICKER",
        date_column: str = "DATE",
    ) -> str:
        """Build a validation query filtered by ticker/date key slices.

        Parameters
        ----------
        table_name
            Table to query.
        ticker_values
            Distinct ticker values used for `IN` filtering.
        date_values
            Distinct date values (YYYY-MM-DD strings) used for date filtering.
        schema
            SQL schema containing ``table_name``.
        ticker_column
            Ticker column name in the target table.
        date_column
            Date column name in the target table.

        Returns
        -------
        str
            Rendered SQL statement.
        """
        ticker_filter = self.add_in_filter(
            self.quote_ident(ticker_column), ticker_values
        )
        date_filter = self.add_in_filter(
            f"CAST({self.quote_ident(date_column)} AS date)",
            date_values,
        )
        return self.render_sql_query(
            query_path=self.resolve_sql_path("validate_ticker_date.txt"),
            filters={
                "schema": self.quote_ident(schema),
                "table_name": self.quote_ident(table_name),
                "ticker_filter": ticker_filter,
                "date_filter": date_filter,
            },
        )


default_sql_client = SQLClient()

__all__ = ["SQLClient", "default_sql_client", "DateLike"]
