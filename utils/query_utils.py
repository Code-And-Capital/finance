from pathlib import Path
import re
from typing import Dict


_PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def render_sql_query(
    *,
    query_path: str | Path,
    filters: Dict[str, str],
) -> str:
    """
    Read a SQL query from query_path and replace placeholders like
    {date_filter}, {ticker_filter}, etc. with values provided in `filters`.

    - All placeholders found in the query MUST be present in `filters`
    - Extra filters in `filters` are ignored
    - Values are assumed to already be valid SQL fragments
    """

    query_path = Path(query_path)

    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    sql = query_path.read_text()

    # Find all placeholders in the query
    required_filters = set(_PLACEHOLDER_PATTERN.findall(sql))

    # Check that all required filters are provided
    missing = required_filters - filters.keys()
    if missing:
        raise ValueError(
            f"Missing required SQL filters: {sorted(missing)} "
            f"for query {query_path.name}"
        )

    # Replace only placeholders that appear in the query
    for name in required_filters:
        sql = sql.replace(f"{{{name}}}", filters[name])

    return sql
