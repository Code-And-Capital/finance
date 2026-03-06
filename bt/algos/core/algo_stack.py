from __future__ import annotations

from typing import Any

from .algo import Algo


class AlgoStack(Algo):
    """Execute multiple algos in sequence with short-circuit semantics.

    Normal mode:
    - Stop at first ``False`` and return ``False``.

    ``run_always`` mode:
    - If any algo in the stack has ``run_always=True``, evaluation still tracks
      the first failure for the returned result, but algos flagged with
      ``run_always`` continue to execute after a prior failure.
    """

    def __init__(self, *algos: Algo) -> None:
        """Initialize stack with ordered algo callables."""
        super().__init__()
        self.algos = algos
        self.check_run_always = any(
            bool(getattr(algo, "run_always", False)) for algo in self.algos
        )

    def __call__(self, target: Any) -> bool:
        """Run stack members against ``target`` and return aggregate success."""
        if not self.check_run_always:
            return all(algo(target) for algo in self.algos)

        res = True
        for algo in self.algos:
            if res:
                res = algo(target)
            elif bool(getattr(algo, "run_always", False)):
                algo(target)
        return res
