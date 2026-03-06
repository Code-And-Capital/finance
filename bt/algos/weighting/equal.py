from __future__ import annotations

from typing import Any

import pandas as pd

from bt.algos.core import Algo


class WeighEqually(Algo):
    """Assign equal weights across selected assets.

    This weighting algo follows a two-step optimizer-style flow:
    1. ``set_problem`` captures current selected assets.
    2. ``solve_problem`` computes equal allocations.

    The computed weights are written to ``target.temp['weights']``.
    """

    def __init__(self) -> None:
        """Initialize equal-weighting state containers."""
        super().__init__()
        self.universe: list[Any] = []
        self.n: int = 0
        self.allocations: dict[Any, float] = {}
        self.allocation_history: pd.DataFrame = pd.DataFrame()

    def set_problem(self, universe: list[Any]) -> None:
        """Set current selected universe for equal-weight allocation."""
        if not isinstance(universe, list):
            raise TypeError("WeighEqually `universe` must be a list.")
        self.universe = list(dict.fromkeys(universe))
        self.n = len(self.universe)

    def solve_problem(self) -> None:
        """Solve equal-weight allocation into ``self.allocations``."""
        if self.n == 0:
            self.allocations = {}
            return

        w = 1.0 / self.n
        self.allocations = {asset: w for asset in self.universe}

    def __call__(self, target) -> bool:
        """Compute and store equal weights into ``target.temp['weights']``."""
        temp = self._resolve_temp(target)
        if temp is None:
            return False

        selected_raw = temp.get("selected", [])
        if not isinstance(selected_raw, list):
            return False

        self.set_problem(selected_raw)
        self.solve_problem()
        temp["weights"] = self.allocations

        now = self._resolve_now(target)
        if now is not None and self.allocations:
            self.allocation_history.loc[now, list(self.allocations.keys())] = list(
                self.allocations.values()
            )
        return True
