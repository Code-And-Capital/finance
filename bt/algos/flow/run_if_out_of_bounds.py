from typing import Any

from bt.algos.core import Algo
from utils.math_utils import validate_non_negative


class RunIfOutOfBounds(Algo):
    """Trigger when asset weights drift outside a tolerance threshold.

    Parameters
    ----------
    tolerance : float
        Non-negative drift threshold.
    mode : str, optional
        Drift mode. ``"relative"`` computes ``abs((current-target)/target)``
        for non-zero targets (and ``abs(current)`` when target is zero).
        ``"absolute"`` computes ``abs(current-target)``.

    Notes
    -----
    This class only evaluates non-cash entries from ``target.temp["weights"]``.
    Use :class:`RunIfCashOutOfBounds` for cash drift checks.
    """

    def __init__(self, tolerance: float, mode: str = "relative"):
        """Initialize drift-triggered rebalance condition.

        Parameters
        ----------
        tolerance : float
            Non-negative drift threshold.
        mode : str, optional
            Drift metric mode (``"relative"`` or ``"absolute"``).
        Raises
        ------
        ValueError
            If ``tolerance`` is negative, or if mode is invalid.
        """
        super().__init__()
        if mode not in {"relative", "absolute"}:
            raise ValueError(
                "RunIfOutOfBounds `mode` must be 'relative' or 'absolute'."
            )

        self.tolerance = validate_non_negative(
            tolerance, "RunIfOutOfBounds `tolerance`"
        )
        self.mode = mode

    def _drift(self, current: float, target: float) -> float:
        if self.mode == "absolute":
            return abs(current - target)
        if target == 0:
            return abs(current)
        return abs((current - target) / target)

    def _is_outside(self, drift: float) -> bool:
        return drift > self.tolerance

    def _weight_items(self, weight_mapping: Any):
        if hasattr(weight_mapping, "items"):
            return list(weight_mapping.items())
        return None

    def __call__(self, target: Any) -> bool:
        """Evaluate whether target allocations are out of bounds.

        Parameters
        ----------
        target : Any
            Target object expected to expose ``temp`` and ``children``.

        Returns
        -------
        bool
            ``True`` when any tracked allocation exceeds tolerance, otherwise
            ``False``. Returns ``False`` when required state is missing.
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return False

        if "weights" not in temp:
            return False

        items = self._weight_items(temp["weights"])
        if items is None:
            return False

        children = getattr(target, "children", {})
        for cname, desired_weight in items:
            if cname == "cash":
                continue

            current_weight = children[cname].weight if cname in children else 0.0
            drift = self._drift(current_weight, desired_weight)
            if self._is_outside(drift):
                return True

        return False


class RunIfCashOutOfBounds(Algo):
    """Trigger when cash allocation drifts outside a tolerance threshold.

    Parameters
    ----------
    tolerance : float
        Non-negative maximum allowed absolute drift in cash weight.
    """

    def __init__(self, tolerance: float):
        """Initialize cash drift trigger."""
        super().__init__()
        self.tolerance = validate_non_negative(
            tolerance, "RunIfCashOutOfBounds `tolerance`"
        )

    def __call__(self, target: Any) -> bool:
        """Evaluate whether cash weight is outside tolerance.

        Parameters
        ----------
        target : Any
            Target object expected to expose ``temp``, ``capital``, and ``value``.

        Returns
        -------
        bool
            ``True`` when cash drift exceeds tolerance, else ``False``.
            Returns ``False`` when required state is missing.
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return False

        if "cash" not in temp:
            return False

        total_value = getattr(target, "value", 0.0)
        if total_value == 0:
            return False

        current_cash_weight = getattr(target, "capital", 0.0) / total_value
        desired_cash_weight = temp["cash"]
        drift = abs(current_cash_weight - desired_cash_weight)
        return drift > self.tolerance
