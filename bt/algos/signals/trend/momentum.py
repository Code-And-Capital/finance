from bt.algos.core import AlgoStack
from bt.algos.selection.ranking import SelectN
from bt.algos.factors import TotalReturn
import pandas as pd
from utils.math_utils import validate_integer


class MomentumSignal(AlgoStack):
    """Select top names by trailing total return.

    Internally this is an ``AlgoStack``:
    1. :class:`TotalReturn` computes returns into ``temp["total_return"]``
    2. :class:`SelectN` selects top/bottom names from that metric

    Parameters
    ----------
    n : int
        Number of names to select. Must be a positive integer.
    lookback : pandas.DateOffset, optional
        Return lookback window used by ``TotalReturn``.
    lag : pandas.DateOffset, optional
        Lag applied by ``TotalReturn`` to avoid look-ahead.
    sort_descending : bool, optional
        If ``True`` (default), highest returns rank first.
    """

    def __init__(
        self,
        n: int,
        lookback: pd.DateOffset = pd.DateOffset(months=3),
        lag: pd.DateOffset = pd.DateOffset(days=0),
        sort_descending: bool = True,
    ) -> None:
        """Initialize momentum signal stack."""
        n_val = int(validate_integer(n, "MomentumSignal `n`"))
        if n_val <= 0:
            raise ValueError("MomentumSignal `n` must be > 0.")
        if not isinstance(lookback, pd.DateOffset):
            raise TypeError("MomentumSignal `lookback` must be a pandas.DateOffset.")
        if not isinstance(lag, pd.DateOffset):
            raise TypeError("MomentumSignal `lag` must be a pandas.DateOffset.")
        if not isinstance(sort_descending, bool):
            raise TypeError("MomentumSignal `sort_descending` must be a bool.")

        super().__init__(
            TotalReturn(lookback=lookback, lag=lag),
            SelectN(
                n=n_val,
                sort_descending=sort_descending,
                stat_key="total_return",
            ),
        )

    def __call__(self, target) -> bool:
        """Ensure candidate pool exists, then run momentum stack."""
        context = self._resolve_temp_universe_now(target)
        if context is None:
            return False
        temp, universe, _ = context

        selected = self._resolve_candidate_pool_with_fallback(
            temp,
            lambda: temp.__setitem__("selected", list(universe.columns)) or True,
            allowed_candidates=list(universe.columns),
        )
        if selected is None:
            return False
        temp["selected"] = selected

        return super().__call__(target)
