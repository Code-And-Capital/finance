from bt.core.algo_base import Algo
from typing import Optional


class PrintDate(Algo):
    """
    Algo that prints the current timestamp contained in the target context.

    This algorithm is primarily intended for debugging inside a strategy run.
    It can help verify that the backtest engine is iterating through dates
    correctly, or ensure that certain events are being triggered at expected
    times.

    Attributes
    ----------
    prefix : str, optional
        A custom prefix printed before the date output. Useful for tagging
        log lines when multiple debugging algos are in use.
    """

    def __init__(self, prefix: str | None = None):
        """
        Initialize the PrintDate debugging algo.

        Parameters
        ----------
        prefix : str, optional
            Optional text printed before the date. If not provided,
            only the timestamp will be printed.
        """
        self.prefix = prefix

    def __call__(self, target) -> bool:
        """
        Execute the algo and print the current date.

        Parameters
        ----------
        target : bt.backtest.Target
            The target context provided by the backtest engine. Must contain
            a `now` attribute representing the current simulation timestamp.

        Returns
        -------
        bool
            Always returns True so the algo chain continues.
        """
        if self.prefix:
            print(f"{self.prefix} {target.now}")
        else:
            print(target.now)

        return True


class PrintTempData(Algo):
    """
    Algo that prints the temporary data (`target.temp`) stored during a
    backtest run.

    This is primarily used for debugging and introspection. The algorithm
    allows formatted output to inspect specific keys or structures stored
    in the temporary dictionary.

    Examples
    --------
    Print the entire temp dictionary:
        PrintTempData()

    Print a specific value:
        PrintTempData("current_vol={current_vol}")

    Attributes
    ----------
    fmt_string : str or None
        Optional string format pattern. If provided, it is formatted using
        the contents of ``target.temp`` via ``fmt_string.format(**target.temp)``.
    """

    def __init__(self, fmt_string: Optional[str] = None):
        """
        Initialize the PrintTempData debugging algo.

        Parameters
        ----------
        fmt_string : str, optional
            A Python format string referencing keys inside ``target.temp``.
            Example: ``"Momentum: {momentum_value}"``.
            If None, the entire ``target.temp`` dictionary will be printed.
        """
        super().__init__()
        self.fmt_string = fmt_string

    def __call__(self, target) -> bool:
        """
        Execute the algorithm and print temporary data.

        Parameters
        ----------
        target : bt.backtest.Target
            The backtest target context containing a ``temp`` dictionary
            that stores short-lived, intermediate values during strategy
            evaluation.

        Returns
        -------
        bool
            Always returns True so that the algo chain continues.

        Notes
        -----
        If ``fmt_string`` references keys that do not exist in ``target.temp``,
        a KeyError will be caught and displayed, preventing the entire algo
        chain from failing.
        """
        if self.fmt_string is None:
            print(target.temp)
            return True

        try:
            print(self.fmt_string.format(**target.temp))
        except KeyError as exc:
            missing_key = exc.args[0]
            print(f"[PrintTempData] Missing key in target.temp: '{missing_key}'")
        except Exception as exc:
            print(f"[PrintTempData] Formatting error: {exc}")

        return True


class PrintInfo(Algo):
    """
    Debugging Algo that prints formatted information from the ``target``
    strategy or node object.

    The formatting string is interpolated with attributes from the target's
    ``__dict__``. This allows inspection of internal state such as ``name``,
    ``now``, or any custom variables stored on the strategy object.

    Examples
    --------
    Print basic info (default):
        PrintInfo()

    Print the strategy name and current date:
        PrintInfo("Strategy {name} at {now}")

    Print several pieces of metadata:
        PrintInfo("Node {name}: Children={children}  Temp={temp}")

    Attributes
    ----------
    fmt_string : str
        A Python format string that will be interpolated with keys from
        ``target.__dict__`` using ``fmt_string.format(**target.__dict__)``.
    """

    def __init__(self, fmt_string: str = "{name} {now}"):
        """
        Initialize the PrintInfo debugging algo.

        Parameters
        ----------
        fmt_string : str, optional
            A format string referencing attributes of the target node or strategy.
            Defaults to ``"{name} {now}"``. If a referenced attribute does not
            exist in ``target.__dict__``, a clear warning is printed.
        """
        super().__init__()
        self.fmt_string = fmt_string

    def __call__(self, target) -> bool:
        """
        Execute the algo and print the formatted target information.

        Parameters
        ----------
        target : bt.backtest.Target or bt.core.Node
            An object whose attributes will be used for formatting. Typically
            a Strategy, Node, or Target instance within the backtest engine.

        Returns
        -------
        bool
            Always ``True`` so that the algo chain continues.

        Notes
        -----
        Any missing attributes referenced in the format string will be
        reported rather than causing the algo chain to crash.
        """
        try:
            print(self.fmt_string.format(**target.__dict__))
        except KeyError as exc:
            missing_key = exc.args[0]
            print(
                f"[PrintInfo] Missing attribute in target.__dict__: '{missing_key}' "
                f"for fmt_string='{self.fmt_string}'"
            )
        except Exception as exc:
            print(f"[PrintInfo] Formatting error: {exc}")

        return True


class PrintRisk(Algo):
    """
    Algo that prints the risk attributes of a target object.

    This class is a simple utility for inspecting the risk data associated
    with a target in a backtesting context. The printed output can be either
    the entire risk dictionary or a custom formatted string.

    Args:
        fmt_string (str, optional): A format string to display specific risk
            attributes. Placeholders (e.g., `{volatility}`, `{drawdown}`)
            should correspond to keys in the target's `risk` dictionary.
            If not provided, the entire risk dictionary will be printed.

    Example:
        >>> algo = PrintRisk(fmt_string="Volatility: {volatility}, Drawdown: {max_drawdown}")
        >>> algo(target)
    """

    def __init__(self, fmt_string: str = "") -> None:
        """
        Initializes the PrintRisk Algo.

        Args:
            fmt_string (str, optional): A format string to specify which
                risk attributes to print. Defaults to an empty string, which
                prints the full risk dictionary.
        """
        super().__init__()
        self.fmt_string: str = fmt_string

    def __call__(self, target) -> bool:
        """
        Prints the risk data of the given target.

        If `fmt_string` was provided, prints the formatted risk attributes
        using that string. Otherwise, prints the entire `risk` dictionary
        of the target.

        Args:
            target (Any): The object whose risk attributes will be printed.
                The object must have a `risk` attribute that is a dictionary.

        Returns:
            bool: Always returns True.
        """
        if hasattr(target, "risk"):
            if self.fmt_string:
                print(self.fmt_string.format(**target.risk))
            else:
                print(target.risk)
        return True
