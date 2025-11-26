import re
from bt.core import Algo


class SelectRegex(Algo):
    """
    Filters the current selection based on a regular expression.

    This algorithm looks at the tickers in temp['selected'] and
    keeps only those whose names match the given regex pattern.

    Args:
        regex (str): Regular expression applied to ticker names.

    Sets:
        selected

    Requires:
        selected
    """

    def __init__(self, regex: str) -> None:
        """
        Initialize the regex selector.

        Parameters:
            regex (str): Regular expression pattern.
        """
        super().__init__()
        self.regex = re.compile(regex)

    def __call__(self, target) -> bool:
        """
        Apply the regex filter to temp['selected'].

        Parameters:
            target: Strategy/backtest container providing temp and universe.

        Returns:
            True
        """
        selected = target.temp["selected"]
        filtered = [s for s in selected if self.regex.search(s)]
        target.temp["selected"] = filtered
        return True
