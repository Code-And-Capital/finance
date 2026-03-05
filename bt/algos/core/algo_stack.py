from __future__ import annotations

from typing import Any

from .algo import Algo


def run_always(func):
    """Decorator to force an algo to run even after prior stack failures."""
    func.run_always = True
    return func


class AlgoStack(Algo):
    """Run multiple algos sequentially until a failure is encountered."""

    def __init__(self, *algos: Algo) -> None:
        super().__init__()
        self.algos = algos
        self.check_run_always = any(hasattr(algo, "run_always") for algo in self.algos)

    def __call__(self, target: Any) -> bool:
        """Execute stack members on ``target`` in order."""
        if not self.check_run_always:
            return all(algo(target) for algo in self.algos)

        res = True
        for algo in self.algos:
            if res:
                res = algo(target)
            elif getattr(algo, "run_always", False):
                algo(target)
        return res
