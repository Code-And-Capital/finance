from typing import Any


def run_always(func):
    """
    Decorator for Algo methods to ensure they run on every
    AlgoStack execution, regardless of earlier failures.

    Usage:
        @run_always
        def my_algo(target):
            ...

    Parameters
    ----------
    func : callable
        The Algo function to decorate.

    Returns
    -------
    callable
        The same function with a `run_always = True` attribute.
    """
    func.run_always = True
    return func


class Algo:
    """
    Base class for strategy Algos.

    Algos modularize strategy logic, making it composable, testable, and
    maintainable. They are designed to follow the Unix philosophy: do one
    thing well.

    Algos are callable objects (or functions) that receive one argument,
    typically the strategy instance (referred to as `target`), and return
    a boolean indicating success or whether a certain condition is met.

    This base class supports stateful Algos via instance variables. Stateless
    logic can be implemented as plain functions.

    Parameters
    ----------
    name : str, optional
        Name of the Algo. Defaults to the class name if not provided.
    """

    def __init__(self, name: str | None = None) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """
        Algo name. Defaults to class name if not explicitly set.
        """
        if self._name is None:
            self._name = self.__class__.__name__
        return self._name

    def __call__(self, target: Any) -> bool:
        """
        Execute the Algo logic on the target (strategy).

        Parameters
        ----------
        target : Any
            The strategy or object on which the Algo operates.

        Returns
        -------
        bool
            Typically indicates success or whether a condition was satisfied.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses.
        """
        raise NotImplementedError(f"{self.name} not implemented!")


class AlgoStack(Algo):
    """
    An AlgoStack runs multiple Algos in sequence until a failure is encountered.

    This allows grouping a set of Algos together. Each Algo in the stack is
    executed in order. Execution stops if an Algo returns False, unless an
    Algo has a `run_always` attribute, in which case it will run even if a
    prior Algo failed.

    Parameters
    ----------
    *algos : Algo
        A variable number of Algo instances to run sequentially.
    """

    def __init__(self, *algos: Algo) -> None:
        super().__init__()
        self.algos = algos
        self.check_run_always = any(hasattr(algo, "run_always") for algo in self.algos)

    def __call__(self, target: Any) -> bool:
        """
        Execute the stack of Algos on the given target.

        Parameters
        ----------
        target : Any
            The strategy or object on which the Algos operate.

        Returns
        -------
        bool
            True if all Algos (that do not have `run_always`) succeed,
            otherwise False.
        """
        if not self.check_run_always:
            # Normal execution: stop at first failure
            return all(algo(target) for algo in self.algos)

        # Execution when at least one Algo has `run_always`
        res = True
        for algo in self.algos:
            if res:
                res = algo(target)
            elif getattr(algo, "run_always", False):
                algo(target)
        return res
