"""Utility algos for covariance transformations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from bt.algos.core import Algo
from utils.math_utils import validate_non_negative, validate_real


class AnnualizeCovariance(Algo):
    """Scale ``temp['covariance']`` by an annualization factor.

    Parameters
    ----------
    annualization_factor : float, optional
        Multiplicative factor used to annualize covariance values.
        A common daily-data value is ``252``.

    Notes
    -----
    This algo expects ``temp["covariance"]`` to already exist as a
    ``pandas.DataFrame``.
    """

    def __init__(self, annualization_factor: float = 252.0) -> None:
        super().__init__()
        self.annualization_factor = validate_non_negative(
            validate_real(
                annualization_factor,
                "AnnualizeCovariance `annualization_factor`",
            ),
            "AnnualizeCovariance `annualization_factor`",
        )

    def __call__(self, target: Any) -> bool:
        """Apply annualization to ``target.temp['covariance']``.

        Parameters
        ----------
        target : Any
            Strategy-like object exposing ``temp``.

        Returns
        -------
        bool
            ``True`` when covariance is present and scaled successfully,
            otherwise ``False``.
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return False

        covariance = temp.get("covariance")
        if not isinstance(covariance, pd.DataFrame):
            return False

        temp["covariance"] = covariance * self.annualization_factor
        return True
