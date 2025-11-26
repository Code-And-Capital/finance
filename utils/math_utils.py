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
