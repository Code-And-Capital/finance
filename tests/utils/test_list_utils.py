import pytest

from utils.list_utils import normalize_string_list, normalize_to_list


def test_normalize_to_list_none():
    assert normalize_to_list(None) is None


def test_normalize_to_list_string():
    result = normalize_to_list("AAPL")
    assert result == ["AAPL"]


def test_normalize_to_list_list():
    result = normalize_to_list(["AAPL", "MSFT"])
    assert result == ["AAPL", "MSFT"]


def test_normalize_to_list_tuple():
    result = normalize_to_list(("AAPL", "MSFT"))
    assert result == ["AAPL", "MSFT"]


def test_normalize_to_list_invalid_type_raises():
    with pytest.raises(TypeError) as excinfo:
        normalize_to_list(123)

    assert "Input must be a string or a sequence of strings" in str(excinfo.value)


def test_normalize_string_list_none():
    assert normalize_string_list(None) is None


def test_normalize_string_list_strips_whitespace():
    assert normalize_string_list([" AAPL ", "MSFT"]) == ["AAPL", "MSFT"]


def test_normalize_string_list_rejects_non_strings():
    with pytest.raises(TypeError, match="tickers must contain only strings"):
        normalize_string_list(["AAPL", 1], field_name="tickers")


def test_normalize_string_list_rejects_empty_strings():
    with pytest.raises(ValueError, match="tickers must not contain empty strings"):
        normalize_string_list(["AAPL", " "], field_name="tickers")
