from bt.core import Algo


class Require(Algo):
    """
    Flow control Algo that evaluates a predicate on a specific item in temp.

    This Algo is useful for controlling flow based on the presence or value
    of items in `target.temp`. It returns the result of a predicate function
    applied to a specified item, or a default value if the item is missing or None.

    Example:
        >>> pred = lambda x: len(x) > 0
        >>> require_algo = Require(pred=pred, item='selected', if_none=False)
        >>> require_algo(target)  # Returns True if 'selected' exists and len > 0

    Args:
        pred (Callable[[Any], bool]): A function that takes the item from temp
            and returns a boolean. Can be a simple lambda or a more complex function.
        item (str): The key in `target.temp` to evaluate.
        if_none (bool, optional): Value to return if the item is missing or None.
            Defaults to False.
    """

    def __init__(self, pred, item: str, if_none: bool = False) -> None:
        """
        Initializes the Require Algo.

        Args:
            pred (Callable[[Any], bool]): Predicate function to apply to the item.
            item (str): Key in `target.temp` to check.
            if_none (bool, optional): Return value if item is missing or None.
                Defaults to False.
        """
        super().__init__()
        self.item = item
        self.pred = pred
        self.if_none = if_none

    def __call__(self, target) -> bool:
        """
        Evaluates the predicate on the specified item in target.temp.

        Args:
            target (Any): The object containing a `temp` attribute (dict-like).

        Returns:
            bool: Result of `pred(item)` if present, otherwise `if_none`.
        """
        if self.item not in target.temp:
            return self.if_none

        item_value = target.temp[self.item]

        if item_value is None:
            return self.if_none

        return self.pred(item_value)
