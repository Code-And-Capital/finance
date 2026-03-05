from bt.algos.core import Algo
from typing import Any, Callable


class Require(Algo):
    """Require a predicate to hold for a value in ``target.temp``.

    Parameters
    ----------
    pred : Callable[[Any], bool]
        Predicate evaluated on ``target.temp[item]`` when present.
    item : str
        Key to read from ``target.temp``.
    if_none : bool, optional
        Return value when ``temp`` is missing, key is absent, or value is ``None``.
    """

    def __init__(
        self, pred: Callable[[Any], bool], item: str, if_none: bool = False
    ) -> None:
        """Initialize the predicate-gated flow control.

        Raises
        ------
        TypeError
            If ``pred`` is not callable.
        """
        super().__init__()
        if not callable(pred):
            raise TypeError("Require `pred` must be callable.")
        self.item = item
        self.pred = pred
        self.if_none = bool(if_none)

    def __call__(self, target: Any) -> bool:
        """Evaluate requirement on current target.

        Parameters
        ----------
        target : Any
            Target object expected to expose a dict-like ``temp``.

        Returns
        -------
        bool
            Predicate result when item value exists, otherwise ``if_none``.
        """
        temp = self._resolve_temp(target)
        if temp is None:
            return self.if_none

        if self.item not in temp:
            return self.if_none

        item_value = temp[self.item]

        if item_value is None:
            return self.if_none

        return bool(self.pred(item_value))
