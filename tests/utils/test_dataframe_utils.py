import pandas as pd
import numpy as np
import pytest

from utils.dataframe_utils import (
    convert_columns_to_numeric,
    df_to_dict,
    normalize_columns,
    add_missing_tickers,
)


def test_all_numeric_columns():
    df = pd.DataFrame(
        {
            "A": ["1", "2", "3"],
            "B": [4, 5, 6],
        }
    )
    result = convert_columns_to_numeric(df.copy())

    assert result["A"].dtype.kind in "if"
    assert result["B"].dtype.kind in "if"
    assert result.equals(pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}))


def test_non_numeric_column_remains():
    df = pd.DataFrame(
        {
            "A": ["x", "y", "z"],
            "B": ["1", "2", "3"],
        }
    )
    result = convert_columns_to_numeric(df.copy())

    # B should convert
    assert result["B"].dtype.kind in "if"

    # A should remain unchanged
    assert result["A"].equals(df["A"])


def test_mixed_numeric_and_non_numeric_values():
    df = pd.DataFrame(
        {
            "A": ["1", "2", "bad"],
        }
    )
    result = convert_columns_to_numeric(df.copy())

    assert result["A"].dtype == object
    assert result["A"].equals(df["A"])


def test_numeric_with_commas_not_convertible():
    df = pd.DataFrame({"A": ["1,000", "2,000"]})

    # pd.to_numeric("1,000") raises ValueError, so this column should remain unchanged
    result = convert_columns_to_numeric(df.copy())

    assert result["A"].tolist() == ["1,000", "2,000"]
    assert result["A"].dtype == object


def test_column_with_none_values():
    df = pd.DataFrame({"A": [1, None, "3"]})
    result = convert_columns_to_numeric(df.copy())

    assert result["A"].dtype.kind in "if"


def test_empty_dataframe():
    df = pd.DataFrame()
    result = convert_columns_to_numeric(df.copy())
    assert result.empty
    assert list(result.columns) == []


def test_mixed_column_types():
    df = pd.DataFrame(
        {
            "A": ["1", 2, 3.0],
            "B": ["x", 1, 2],
        }
    )
    result = convert_columns_to_numeric(df.copy())

    # A fully numeric
    assert result["A"].dtype.kind in "if"

    # B should coerce x to NaN
    assert result["B"].equals(df["B"])


def test_basic_conversion():
    df = pd.DataFrame({"key": ["A", "B", "C"], "value": [1, 2, 3]})
    result = df_to_dict(df, "key", "value")
    assert result == {"A": 1, "B": 2, "C": 3}


def test_duplicate_keys_overwrite():
    df = pd.DataFrame({"key": ["A", "A", "B"], "value": [1, 2, 3]})
    result = df_to_dict(df, "key", "value")

    # Pandas set_index keeps the last occurrence as the dict value
    assert result == {"A": 2, "B": 3}


def test_missing_key_column():
    df = pd.DataFrame({"value": [1, 2]})
    with pytest.raises(KeyError):
        df_to_dict(df, "missing", "value")


def test_missing_value_column():
    df = pd.DataFrame({"key": ["A", "B"]})
    with pytest.raises(KeyError):
        df_to_dict(df, "key", "missing")


def test_null_keys_allowed():
    df = pd.DataFrame({"key": ["A", None, "C"], "value": [1, 2, 3]})
    result = df_to_dict(df, "key", "value")

    # None becomes a valid dict key
    assert result == {"A": 1, None: 2, "C": 3}


def test_null_values():
    df = pd.DataFrame({"key": ["A", "B", "C"], "value": [1, None, 3]})
    result = df_to_dict(df, "key", "value")

    assert result["A"] == 1
    assert np.isnan(result["B"])
    assert result["C"] == 3
    assert set(result.keys()) == {"A", "B", "C"}


def test_mixed_type_keys():
    df = pd.DataFrame({"key": ["A", 100, 3.14], "value": [1, 2, 3]})
    result = df_to_dict(df, "key", "value")
    assert result == {"A": 1, 100: 2, 3.14: 3}


def test_empty_dataframe():
    df = pd.DataFrame(columns=["key", "value"])
    result = df_to_dict(df, "key", "value")
    assert result == {}


def test_non_string_column_names():
    df = pd.DataFrame({0: ["A", "B"], 1: [10, 20]})
    result = df_to_dict(df, 0, 1)
    assert result == {"A": 10, "B": 20}


def test_multiindex_columns_still_work():
    df = pd.DataFrame({("col", "key"): ["A", "B"], ("col", "value"): [1, 2]})
    result = df_to_dict(df, ("col", "key"), ("col", "value"))
    assert result == {"A": 1, "B": 2}


def test_basic_normalization():
    df = pd.DataFrame(columns=["col one", "colTwo", "COL three"])
    result = normalize_columns(df.copy())

    assert result.columns.tolist() == ["COL_ONE", "COLTWO", "COL_THREE"]


def test_no_spaces_only_uppercase():
    df = pd.DataFrame(columns=["abc", "Def", "GHI"])
    result = normalize_columns(df.copy())

    assert result.columns.tolist() == ["ABC", "DEF", "GHI"]


def test_special_characters_remain():
    df = pd.DataFrame(columns=["col-name", "col/name", "col.name"])
    result = normalize_columns(df.copy())

    assert result.columns.tolist() == ["COL-NAME", "COL/NAME", "COL.NAME"]


def test_numeric_column_names():
    df = pd.DataFrame(columns=[1, 2, 3])
    result = normalize_columns(df.copy())

    # Numeric names converted to strings then uppercased (no change)
    assert result.columns.tolist() == ["1", "2", "3"]


def test_mixed_types_in_columns():
    df = pd.DataFrame(columns=["name", 123, None])
    result = normalize_columns(df.copy())

    # None becomes "NONE", numbers become strings
    assert result.columns.tolist() == ["NAME", "123", "NONE"]


def test_duplicate_columns_after_normalization():
    df = pd.DataFrame(columns=["col one", "COL_ONE"])
    result = normalize_columns(df.copy())

    # Pandas will allow duplicate column names
    assert result.columns.tolist() == ["COL_ONE", "COL_ONE"]


def test_empty_dataframe():
    df = pd.DataFrame()
    result = normalize_columns(df.copy())

    assert list(result.columns) == []


def test_leading_trailing_spaces():
    df = pd.DataFrame(columns=["  col  one  ", " col two"])
    result = normalize_columns(df.copy())

    # Leading/trailing spaces become underscores as well
    assert result.columns.tolist() == ["__COL__ONE__", "_COL_TWO"]


def test_add_missing_tickers_adds_missing_rows():
    df = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "START_DATE": ["2010-01-01", "2011-01-01"],
        }
    )

    result = add_missing_tickers(df, ["AAPL", "MSFT", "GOOG"])

    assert set(result["TICKER"]) == {"AAPL", "MSFT", "GOOG"}
    assert result.loc[result["TICKER"] == "GOOG", "START_DATE"].iloc[0] == "2000-01-01"


def test_add_missing_tickers_no_change_when_all_present():
    df = pd.DataFrame(
        {
            "TICKER": ["AAPL", "MSFT"],
            "START_DATE": ["2010-01-01", "2011-01-01"],
        }
    )

    result = add_missing_tickers(df, ["AAPL", "MSFT"])

    assert result.equals(df)


def test_add_missing_tickers_empty_ticker_list():
    df = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "START_DATE": ["2010-01-01"],
        }
    )

    result = add_missing_tickers(df, [])

    assert result.equals(df)


def test_add_missing_tickers_preserves_existing_rows():
    df = pd.DataFrame(
        {
            "TICKER": ["AAPL"],
            "START_DATE": ["2010-01-01"],
            "WEIGHT": [0.05],
        }
    )

    result = add_missing_tickers(df, ["AAPL", "MSFT"])

    assert "WEIGHT" in result.columns
    assert result.loc[result["TICKER"] == "AAPL", "WEIGHT"].iloc[0] == 0.05


def test_add_missing_tickers_missing_ticker_column_raises():
    df = pd.DataFrame(
        {
            "START_DATE": ["2010-01-01"],
        }
    )

    with pytest.raises(KeyError):
        add_missing_tickers(df, ["AAPL"])
