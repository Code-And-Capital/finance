from decimal import Decimal
from numbers import Integral, Real


def is_zero(x, tol: float = 1e-12) -> bool:
    """
    Return True if ``x`` is effectively zero within a numerical tolerance.

    Floating-point rounding and accumulated arithmetic error can cause values
    that should mathematically be zero to appear as very small nonzero numbers.
    This helper provides a consistent, explicit way to check for "practical"
    zeros throughout the framework.

    Parameters
    ----------
    x : float
        The value to test.
    tol : float, optional
        Absolute tolerance within which the value is treated as zero.
        Default is 1e-12, which is typically safe for portfolio calculations.

    Returns
    -------
    bool
        True if ``abs(x) < tol`` else False.
    """
    return abs(x) < tol


def validate_non_negative(value: float, label: str) -> float:
    """
    Return ``value`` as float after enforcing a non-negative constraint.

    Parameters
    ----------
    value : float
        Numeric value to validate.
    label : str
        Field name used in the exception message.

    Returns
    -------
    float
        The validated value converted to ``float``.

    Raises
    ------
    ValueError
        If ``value`` is negative.
    """
    val = float(value)
    if val < 0:
        raise ValueError(f"{label} must be >= 0.")
    return val


def validate_integer(value, label: str) -> int:
    """
    Return ``value`` as ``int`` after enforcing integer type semantics.

    Parameters
    ----------
    value : Any
        Value to validate as an integer.
    label : str
        Field name used in the exception message.

    Returns
    -------
    int
        The validated integer value.

    Raises
    ------
    TypeError
        If ``value`` is not an integer-like numeric type.
    """
    if not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer.")
    return int(value)


def validate_real(value, label: str) -> float:
    """
    Return ``value`` as ``float`` after enforcing real-number semantics.

    Parameters
    ----------
    value : Any
        Value to validate as a real number.
    label : str
        Field name used in the exception message.

    Returns
    -------
    float
        The validated real value.

    Raises
    ------
    TypeError
        If ``value`` is not a real-number type.
    """
    if not isinstance(value, (Real, Decimal)):
        raise TypeError(f"{label} must be numeric.")
    return float(value)
