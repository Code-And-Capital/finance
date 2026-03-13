from typing import Any

import pandas as pd

from bt.algos.core import Algo


class SetFactor(Algo):
    """Set ``target.temp[factor_str]`` from a wide factor DataFrame.

    Parameters
    ----------
    factor_str : str, optional
        Output key written into ``target.temp``. When ``factor_df`` is ``None``,
        this key is also used to resolve the input DataFrame via
        ``target.get_data(factor_str)``.
    factor_df : pandas.DataFrame, optional
        Inline wide DataFrame source. When provided, this source is used
        directly and ``factor_str`` is used only as the output temp key.
    """

    def __init__(
        self,
        factor_str: str = "factor",
        factor_df: pd.DataFrame | None = None,
    ) -> None:
        """Initialize factor-setter algo."""
        super().__init__()

        if not isinstance(factor_str, str) or not factor_str:
            raise TypeError("SetFactor `factor_str` must be a non-empty string.")
        if factor_df is not None and not isinstance(factor_df, pd.DataFrame):
            raise TypeError("SetFactor `factor_df` must be a pandas.DataFrame or None.")

        self.factor_str = factor_str
        self.factor_source = factor_df
        self.factor_source_key: str | None = (
            None if factor_df is not None else factor_str
        )

    def __call__(self, target: Any) -> bool:
        """Resolve factor row at the current market-data timestamp."""
        temp = self._resolve_temp(target)
        if temp is None:
            return False

        now = self._resolve_market_data_now(target)
        if now is None:
            return False

        resolved = self._resolve_wide_data_row_at_now(
            now=now,
            inline_wide=self.factor_source,
            wide_key=self.factor_source_key,
            key_resolver=lambda key: target.get_data(key),
        )
        if resolved is None:
            return False
        _, row = resolved
        temp[self.factor_str] = row
        return True
