import pytest
from pathlib import Path

from utils.query_utils import render_sql_query


def test_render_sql_query_replaces_all_placeholders(tmp_path: Path):
    sql_file = tmp_path / "query.sql"
    sql_file.write_text(
        """
        SELECT *
        FROM company_info
        WHERE 1=1
        {ticker_filter}
        {date_filter}
        """
    )

    result = render_sql_query(
        query_path=sql_file,
        filters={
            "ticker_filter": "AND ticker = 'AAPL'",
            "date_filter": "AND date >= '2023-01-01'",
        },
    )

    assert "{ticker_filter}" not in result
    assert "{date_filter}" not in result
    assert "AND ticker = 'AAPL'" in result
    assert "AND date >= '2023-01-01'" in result


def test_render_sql_query_ignores_extra_filters(tmp_path: Path):
    sql_file = tmp_path / "query.sql"
    sql_file.write_text(
        """
        SELECT *
        FROM LLM
        WHERE 1=1
        {llm_filter}
        """
    )

    result = render_sql_query(
        query_path=sql_file,
        filters={
            "llm_filter": "AND model = 'gpt-4'",
            "date_filter": "AND date >= '2024-01-01'",
        },
    )

    assert "AND model = 'gpt-4'" in result
    assert "date >= '2024-01-01'" not in result


def test_render_sql_query_missing_filter_raises_value_error(tmp_path: Path):
    sql_file = tmp_path / "query.sql"
    sql_file.write_text(
        """
        SELECT *
        FROM company_info
        WHERE 1=1
        {ticker_filter}
        {date_filter}
        """
    )

    with pytest.raises(ValueError) as exc:
        render_sql_query(
            query_path=sql_file,
            filters={
                "ticker_filter": "AND ticker = 'AAPL'",
            },
        )

    message = str(exc.value)
    assert "Missing required SQL filters" in message
    assert "date_filter" in message


def test_render_sql_query_with_no_placeholders(tmp_path: Path):
    sql_file = tmp_path / "query.sql"
    original_sql = """
        SELECT *
        FROM static_table
        WHERE active = 1
    """
    sql_file.write_text(original_sql)

    result = render_sql_query(
        query_path=sql_file,
        filters={
            "unused_filter": "SHOULD_BE_IGNORED",
        },
    )

    assert result.strip() == original_sql.strip()


def test_render_sql_query_file_not_found():
    with pytest.raises(FileNotFoundError):
        render_sql_query(
            query_path="does_not_exist.sql",
            filters={},
        )


def test_render_sql_query_complex_placeholder_names(tmp_path: Path):
    sql_file = tmp_path / "query.sql"
    sql_file.write_text(
        """
        SELECT *
        FROM data
        WHERE 1=1
        {index_filter_1}
        """
    )

    result = render_sql_query(
        query_path=sql_file,
        filters={
            "index_filter_1": "AND idx = 42",
        },
    )

    assert "AND idx = 42" in result
