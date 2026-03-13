"""Company information query helpers."""

from typing import Optional, Sequence

import pandas as pd

from handyman.base import (
    DateLike,
    build_security_filter_sql,
    read_table_by_filters,
    run_sql_template,
)
from sql.script_factory import default_sql_client
from utils.dataframe_utils import ensure_datetime_column
from utils.list_utils import normalize_string_list


def get_company_info(
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return company information with optional ticker/date filters.

    Parameters
    ----------
    tickers
        Optional ticker filter.
    figis
        Optional FIGI filter value(s). Cannot be combined with ``tickers``.
    start_date
        Optional inclusive lower-bound filter for ``DATE``.
    end_date
        Optional inclusive upper-bound filter for ``DATE``.
    configs_path
        Optional configs path for Azure credentials.
    get_latest
        If True, return only the latest row per ticker. In this mode
        ``start_date`` is ignored.

    Returns
    -------
    pandas.DataFrame
        Company information rows with ``DATE`` coerced to datetime.
    """
    security_filter = build_security_filter_sql(
        sql_client=default_sql_client,
        tickers=tickers,
        figis=figis,
    )
    date_filter = (
        ""
        if get_latest
        else "\n".join(
            part
            for part in [
                default_sql_client.add_date_filter("DATE", start_date),
                default_sql_client.add_end_date_filter("DATE", end_date),
            ]
            if part
        )
    )
    sql_file = "company_info_latest.txt" if get_latest else "company_info.txt"

    df = run_sql_template(
        sql_file=sql_file,
        filters={
            "security_filter": security_filter,
            "date_filter": date_filter,
        },
        configs_path=configs_path,
    )
    if "_ROW_NUM" in df.columns:
        df = df.drop(columns=["_ROW_NUM"])

    return ensure_datetime_column(df, "DATE")


def get_officers(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return officer rows with optional ticker/date filters.

    Parameters
    ----------
    tickers
        Optional ticker filter.
    figis
        Optional FIGI filter value(s). Cannot be combined with ``tickers``.
    start_date
        Optional inclusive lower-bound filter for ``DATE``.
    end_date
        Optional inclusive upper-bound filter for ``DATE``.
    configs_path
        Optional configs path for Azure credentials.
    get_latest
        If True, return only the latest officer snapshot per ticker and ignore
        ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Officer rows with ``DATE`` coerced to datetime.
    """
    if not get_latest:
        df = read_table_by_filters(
            table_name="officers",
            tickers=tickers,
            figis=figis,
            start_date=start_date,
            end_date=end_date,
            configs_path=configs_path,
        )
        return ensure_datetime_column(df, "DATE")

    security_filter = build_security_filter_sql(
        sql_client=default_sql_client,
        tickers=tickers,
        figis=figis,
    )
    df = run_sql_template(
        sql_file="officers_latest.txt",
        filters={"security_filter": security_filter, "date_filter": ""},
        configs_path=configs_path,
    )
    return ensure_datetime_column(df, "DATE")


def get_security_master(
    *,
    tickers: Optional[Sequence[str] | str] = None,
    figis: Optional[Sequence[str] | str] = None,
    names: Optional[Sequence[str] | str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    configs_path: Optional[str] = None,
    get_latest: bool = False,
) -> pd.DataFrame:
    """Return security master rows with optional filters.

    Parameters
    ----------
    tickers
        Optional ticker filter.
    figis
        Optional FIGI filter value(s). Cannot be combined with ``tickers``.
    names
        Optional security-name filter applied to ``NAME``.
    start_date
        Optional inclusive lower-bound filter for ``DATE``.
    end_date
        Optional inclusive upper-bound filter for ``DATE``.
    configs_path
        Optional configs path for Azure credentials.
    get_latest
        If True, return only the latest row per ticker and ignore ``start_date``.

    Returns
    -------
    pandas.DataFrame
        Security master rows with ``DATE`` coerced to datetime when present.
    """
    if not get_latest:
        df = read_table_by_filters(
            table_name="security_master",
            tickers=tickers,
            figis=figis,
            start_date=start_date,
            end_date=end_date,
            configs_path=configs_path,
        )
        normalized_names = normalize_string_list(names, field_name="names")
        if normalized_names:
            allowed = {value.strip() for value in normalized_names}
            df = df[df["NAME"].astype(str).str.strip().isin(allowed)]
        return ensure_datetime_column(df, "DATE")

    normalized_names = normalize_string_list(names, field_name="names")
    security_filter = build_security_filter_sql(
        sql_client=default_sql_client,
        tickers=tickers,
        figis=figis,
    )
    name_filter = default_sql_client.add_in_filter("NAME", normalized_names)
    date_filter = ""

    df = run_sql_template(
        sql_file="security_master_latest.txt",
        filters={
            "security_filter": security_filter,
            "name_filter": name_filter,
            "date_filter": date_filter,
        },
        configs_path=configs_path,
    )
    return ensure_datetime_column(df, "DATE")
