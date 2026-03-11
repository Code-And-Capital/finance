"""Expected-returns algos.

This package contains algorithms that compute expected-return vectors for the
current investable universe and store results in
``target.temp["expected_returns"]``.
"""

from .core import ExpectedReturns
from .classic import LogReturn, MedianReturn, SimpleReturn
from .ewma import BlendedExpectedReturn, EWMAExpectedReturns
from .excess import ExcessReturn
from .outliers import TrimmedMeanReturn, WinsorizedMeanReturn
from .realized import RealizedReturn

__all__ = [
    "ExpectedReturns",
    "SimpleReturn",
    "LogReturn",
    "MedianReturn",
    "EWMAExpectedReturns",
    "BlendedExpectedReturn",
    "ExcessReturn",
    "TrimmedMeanReturn",
    "WinsorizedMeanReturn",
    "RealizedReturn",
]
