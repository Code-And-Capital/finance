import pandas as pd
import pytest

from utils.date_utils import coerce_timestamp, coerce_timestamp_or_none, month_index


def test_coerce_timestamp_accepts_valid_inputs():
    assert coerce_timestamp("2024-01-15") == pd.Timestamp("2024-01-15")
    assert coerce_timestamp(pd.Timestamp("2024-01-15")) == pd.Timestamp("2024-01-15")


def test_coerce_timestamp_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="must be a valid timestamp"):
        coerce_timestamp("not-a-date", "date")

    with pytest.raises(ValueError, match="must be a valid timestamp"):
        coerce_timestamp(pd.NaT, "date")


def test_coerce_timestamp_or_none_behavior():
    assert coerce_timestamp_or_none(None) is None
    assert coerce_timestamp_or_none("not-a-date") is None
    assert coerce_timestamp_or_none("2024-01-15") == pd.Timestamp("2024-01-15")


def test_month_index_computes_expected_values():
    assert month_index("2025-12-15") == 2025 * 12 + 12
    assert month_index(pd.Timestamp("2026-01-10")) == 2026 * 12 + 1


def test_month_index_rejects_invalid_values():
    with pytest.raises(ValueError, match="date must be a valid timestamp"):
        month_index("not-a-date", label="date")
