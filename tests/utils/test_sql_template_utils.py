from __future__ import annotations

from pathlib import Path

import pytest

from sql.script_factory import SQLClient


def test_resolve_sql_path_returns_absolute_sql_path():
    client = SQLClient()
    path = client.resolve_sql_path("prices.txt")
    assert Path(path).is_absolute()
    assert path.endswith("sql/templates/prices.txt")


def test_add_in_filter_empty_values_returns_empty_string():
    assert SQLClient.add_in_filter("TICKER", None) == ""
    assert SQLClient.add_in_filter("TICKER", []) == ""


def test_add_in_filter_escapes_single_quotes():
    out = SQLClient.add_in_filter("TICKER", ["BRK.B", "O'REILLY"])
    assert out == "AND TICKER IN ('BRK.B', 'O''REILLY')"


def test_add_date_filter_none_and_valid_values():
    assert SQLClient.add_date_filter("DATE", None) == ""
    assert SQLClient.add_date_filter("DATE", "2024-01-02") == "AND DATE >= '2024-01-02'"


def test_add_date_filter_invalid_raises():
    with pytest.raises(ValueError, match="start_date must be parseable as a date"):
        SQLClient.add_date_filter("DATE", "not-a-date")


def test_add_end_date_filter_none_and_valid_values():
    assert SQLClient.add_end_date_filter("DATE", None) == ""
    assert (
        SQLClient.add_end_date_filter("DATE", "2024-01-31")
        == "AND DATE <= '2024-01-31'"
    )


def test_add_end_date_filter_invalid_raises():
    with pytest.raises(ValueError, match="end_date must be parseable as a date"):
        SQLClient.add_end_date_filter("DATE", "not-a-date")


def test_template_load_render_and_query_state():
    client = SQLClient()

    query = client.render_sql_query(
        query_path=client.resolve_sql_path("prices.txt"),
        filters={"security_filter": "", "date_filter": ""},
    )

    assert "FROM [dbo].[prices]" in query
    assert "{security_filter}" not in query
    assert "{date_filter}" not in query
    assert client.query() == query


def test_build_select_with_filters_query_renders_filters():
    client = SQLClient()
    query = client.build_select_with_filters_query(
        table_name="officers",
        filters_sql=client.add_in_filter("[TICKER]", ["AAPL"]),
    )

    assert "FROM [dbo].[officers]" in query
    assert "WHERE 1 = 1" in query
    assert "AND [TICKER] IN ('AAPL')" in query
