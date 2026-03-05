from bt.algos.core import Algo
from collections.abc import Iterable
from typing import Any


class Not(Algo):
    """Invert the boolean result of another algo.

    Parameters
    ----------
    algo : Algo
        Wrapped algo to evaluate and negate.
    """

    def __init__(self, algo: "Algo") -> None:
        """Initialize negation combinator.

        Raises
        ------
        TypeError
            If ``algo`` is not callable.
        """
        super().__init__()
        if not callable(algo):
            raise TypeError("Not `algo` must be callable.")
        self._algo = algo

    def __call__(self, target: Any) -> bool:
        """Return logical negation of wrapped algo result."""
        return not bool(self._algo(target))


class Or(Algo):
    """Logical OR over a collection of algos.

    Parameters
    ----------
    list_of_algos : Iterable[Algo]
        Iterable of callable algos. Returns ``True`` if any algo is true.
    """

    def __init__(self, list_of_algos) -> None:
        """Initialize OR combinator.

        Raises
        ------
        TypeError
            If ``list_of_algos`` is not iterable or contains non-callables.
        """
        super().__init__()
        if not isinstance(list_of_algos, Iterable):
            raise TypeError("Or `list_of_algos` must be iterable.")
        self._list_of_algos = tuple(list_of_algos)
        if not all(callable(algo) for algo in self._list_of_algos):
            raise TypeError("All items in Or `list_of_algos` must be callable.")

    def __call__(self, target: Any) -> bool:
        """Return ``True`` if any wrapped algo returns truthy."""
        return any(bool(algo(target)) for algo in self._list_of_algos)
