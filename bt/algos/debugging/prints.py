from typing import Any, Optional

from bt.algos.core import Algo
import utils.logging as logging


class PrintDate(Algo):
    """Log the current simulation timestamp from the target.

    Parameters
    ----------
    prefix : str | None, optional
        Optional text prepended to the logged timestamp.

    Notes
    -----
    This algo is intended for diagnostics and always returns ``True`` so it
    does not interrupt an algo stack.
    """

    def __init__(self, prefix: str | None = None):
        """Initialize the date logging algo."""
        self.prefix = prefix

    def __call__(self, target: Any) -> bool:
        """Log ``target.now``.

        Parameters
        ----------
        target : Any
            Target object expected to expose a ``now`` attribute.

        Returns
        -------
        bool
            Always ``True``.
        """
        if not hasattr(target, "now"):
            logging.log("[PrintDate] target is missing required attribute: 'now'")
            return True

        message = f"{self.prefix} {target.now}" if self.prefix else f"{target.now}"
        logging.log(message)

        return True


class PrintTempData(Algo):
    """Log temporary algo state from ``target.temp``.

    Parameters
    ----------
    fmt_string : str | None, optional
        Format string applied as ``fmt_string.format(**target.temp)``.
        If ``None``, the full ``target.temp`` object is logged.
    """

    def __init__(self, fmt_string: Optional[str] = None):
        """Initialize temp-state logger."""
        super().__init__()
        self.fmt_string = fmt_string

    def __call__(self, target: Any) -> bool:
        """Log temporary state for the provided target.

        Parameters
        ----------
        target : Any
            Target object expected to expose a ``temp`` attribute.

        Returns
        -------
        bool
            Always ``True``.
        """
        if not hasattr(target, "temp"):
            logging.log("[PrintTempData] target is missing required attribute: 'temp'")
            return True

        if self.fmt_string is None:
            logging.log(f"{target.temp}")
            return True

        try:
            logging.log(self.fmt_string.format(**target.temp))
        except KeyError as exc:
            missing_key = exc.args[0]
            logging.log(f"[PrintTempData] Missing key in target.temp: '{missing_key}'")
        except Exception as exc:
            logging.log(f"[PrintTempData] Formatting error: {exc}")

        return True


class PrintInfo(Algo):
    """Log formatted attributes from the target object.

    Parameters
    ----------
    fmt_string : str, optional
        Format string interpolated with ``target.__dict__``.
    """

    def __init__(self, fmt_string: str = "{name} {now}"):
        """Initialize target-info logger."""
        super().__init__()
        self.fmt_string = fmt_string

    def __call__(self, target: Any) -> bool:
        """Log formatted info for the provided target.

        Parameters
        ----------
        target : Any
            Target object whose attributes are available for formatting.

        Returns
        -------
        bool
            Always ``True``.
        """
        try:
            logging.log(self.fmt_string.format(**target.__dict__))
        except KeyError as exc:
            missing_key = exc.args[0]
            logging.log(
                f"[PrintInfo] Missing attribute in target.__dict__: '{missing_key}' "
                f"for fmt_string='{self.fmt_string}'"
            )
        except Exception as exc:
            logging.log(f"[PrintInfo] Formatting error: {exc}")

        return True
