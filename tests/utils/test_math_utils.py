from utils.math_utils import (
    is_zero,
    validate_integer,
    validate_non_negative,
    validate_real,
)
import numpy as np
from decimal import Decimal
import pytest


def test_exact_zero():
    assert is_zero(0.0) is True
    assert is_zero(0) is True


def test_small_positive_within_tol():
    assert is_zero(1e-15) is True
    assert is_zero(5e-13, tol=1e-12) is True


def test_small_negative_within_tol():
    assert is_zero(-5e-13, tol=1e-12) is True


def test_exactly_equal_to_tol():
    # abs(x) < tol → equal-to-tolerance should be False
    assert is_zero(1e-12) is False
    assert is_zero(-1e-12) is False


def test_just_outside_tol():
    assert is_zero(1.0000001e-12) is False
    assert is_zero(-1.0000001e-12) is False


def test_larger_values():
    assert is_zero(1e-6) is False
    assert is_zero(-1e-6) is False


def test_non_float_numeric_types():
    assert is_zero(np.float64(0.0))
    assert is_zero(np.float64(1e-15))

    assert is_zero(Decimal("1e-13"), tol=Decimal("1e-12"))
    assert not is_zero(Decimal("1e-4"), tol=Decimal("1e-12"))


def test_custom_tolerance():
    assert is_zero(0.01, tol=0.1) is True
    assert is_zero(0.01, tol=0.001) is False


def test_non_numeric_raises():
    with pytest.raises(TypeError):
        is_zero("0")

    with pytest.raises(TypeError):
        is_zero(None)


def test_validate_non_negative_accepts_zero_and_positive():
    assert validate_non_negative(0, "tol") == 0.0
    assert validate_non_negative(1.5, "tol") == 1.5


def test_validate_non_negative_converts_numeric_like_values():
    assert validate_non_negative("2.5", "tol") == 2.5
    assert validate_non_negative(Decimal("3.0"), "tol") == 3.0


def test_validate_non_negative_raises_for_negative_values():
    with pytest.raises(ValueError, match=r"^tol must be >= 0\.$"):
        validate_non_negative(-0.0001, "tol")


def test_validate_integer_accepts_integer_types():
    assert validate_integer(3, "n") == 3
    assert validate_integer(np.int64(4), "n") == 4
    assert validate_integer(True, "n") == 1


def test_validate_integer_rejects_non_integer_types():
    with pytest.raises(TypeError, match=r"^n must be an integer\.$"):
        validate_integer(1.5, "n")

    with pytest.raises(TypeError, match=r"^n must be an integer\.$"):
        validate_integer("3", "n")


def test_validate_real_accepts_real_types():
    assert validate_real(1, "x") == 1.0
    assert validate_real(np.float64(1.5), "x") == 1.5
    assert validate_real(Decimal("2.25"), "x") == 2.25


def test_validate_real_rejects_non_real_types():
    with pytest.raises(TypeError, match=r"^x must be numeric\.$"):
        validate_real("1.0", "x")
