from bt.core.algo_base import Algo


class Not(Algo):
    """
    Flow control Algo that inverts the result of another Algo.

    This class is useful for "inverting" other flow control Algos. For
    example, you can use `Not(RunAfterDate(...))` or `Not(RunAfterDays(...))`
    to reverse their boolean output.

    Args:
        algo (Algo): The Algo whose return value will be inverted.

    Example:
        >>> from bt.algos import RunAfterDate
        >>> algo = Not(RunAfterDate('2025-12-01'))
        >>> algo(target)
    """

    def __init__(self, algo: "Algo") -> None:
        """
        Initializes the Not Algo.

        Args:
            algo (Algo): The Algo to invert.
        """
        super().__init__()
        self._algo = algo

    def __call__(self, target) -> bool:
        """
        Evaluates the wrapped Algo and returns the inverted result.

        Args:
            target (Any): The object that the wrapped Algo operates on.

        Returns:
            bool: The logical NOT of the wrapped Algo's result.
        """
        return not self._algo(target)


class Or(Algo):
    """
    Flow control Algo that combines multiple Algos using a logical OR.

    This class evaluates a list of Algos and returns True if **any** of
    the contained Algos return True. It is useful for combining multiple
    signals into a single signal.

    Example:
        >>> run_on_date = bt.algos.RunOnDate(pdf.index[0])  # first date in time series
        >>> run_monthly = bt.algos.RunMonthly()
        >>> or_algo = Or([run_monthly, run_on_date])
        >>> or_algo(target)  # Returns True if it's the first date OR 1st of the month

    Args:
        list_of_algos (Iterable[Algo]): A list or iterable of Algos to combine.
            Each Algo is run on the target, and the result is True if **any**
            Algo returns True.
    """

    def __init__(self, list_of_algos) -> None:
        """
        Initializes the Or Algo.

        Args:
            list_of_algos (Iterable[Algo]): List of Algos to evaluate with logical OR.
        """
        super(Or, self).__init__()
        self._list_of_algos = list_of_algos

    def __call__(self, target) -> bool:
        """
        Evaluates the contained Algos and returns True if any Algo returns True.

        Args:
            target (Any): The object that each contained Algo operates on.

        Returns:
            bool: True if any contained Algo returns True, otherwise False.
        """
        res = False
        for algo in self._list_of_algos:
            temp_res = algo(target)
            res = res or temp_res
        return res
