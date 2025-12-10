from utils.math_utils import is_zero
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
