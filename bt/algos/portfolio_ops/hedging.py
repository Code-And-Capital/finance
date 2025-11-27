from typing import List, Union, Optional
import numpy as np
import pandas as pd
from bt.core.algo_base import Algo, StrategyBase


class HedgeRisks(Algo):
    """
    Hedge one or more risk measures using a selected set of hedge instruments.

    This Algo requires that :class:`UpdateRisk` has already been executed
    earlier in the same period so that the ``risk`` attribute is populated
    for the target node (and optional strategy node).

    The Algo operates by:
        1. Reading the current risk vector across the requested measures.
        2. Constructing a Jacobian matrix of unit risks for the chosen hedge
           instruments.
        3. Solving for hedge notionals using either a true inverse or
           the Moore-Penrose pseudo-inverse.
        4. Executing hedging transactions via ``target.transact``.

    Parameters
    ----------
    measures : List[str]
        Names of the risk measures to hedge (e.g., ["IR01", "PVBP"]).
        Must match keys in additional_data["unit_risk"] for all dates.
    pseudo : bool, optional
        If True, uses the pseudo-inverse of the hedge Jacobian, allowing
        rectangular or singular systems. If False, requires a square,
        non-singular Jacobian. Default is False.
    strategy : Optional[StrategyBase], optional
        If provided, the Algo hedges the *combined* risk of both `target`
        and `strategy`. This enables the user to track unhedged strategy
        performance independently from the hedged portfolio. Default: None.
    throw_nan : bool, optional
        If True, raises if any computed hedge amount is NaN. If False,
        silently skips applying that hedge leg. Default: True.

    Requires
    --------
    selected : list
        ``target.temp["selected"]`` must contain the list of hedge instruments.

    Notes
    -----
    - Ensure that :class:`UpdateRisk` has run *before* this Algo.
    - Ensure that the selected hedge instruments appear as columns in the
      corresponding ``unit_risk`` DataFrames.
    """

    def __init__(
        self,
        measures: List[str],
        pseudo: bool = False,
        strategy: Optional[StrategyBase] = None,
        throw_nan: bool = True,
    ) -> None:
        super().__init__(name="HedgeRisks")
        if not measures:
            raise ValueError("Must pass at least one measure to hedge.")

        self.measures = measures
        self.pseudo = pseudo
        self.strategy = strategy
        self.throw_nan = throw_nan

    def _get_risk_value(self, node, measure: str) -> float:
        """
        Return the scalar risk value for a measure on the given node.

        Raises a clear error if UpdateRisk has not initialized the structure.
        """
        if not hasattr(node, "risk"):
            raise ValueError(f"risk not set up on node {node.name}")
        if measure not in node.risk:
            raise ValueError(
                f"risk measure '{measure}' not available on node {node.name}"
            )
        return node.risk[measure]

    def _get_unit_risk(self, security: str, frame: pd.DataFrame, idx: int) -> float:
        """
        Safely retrieve unit risk for a given security at a given index.
        Missing or invalid data defaults to 0.0.
        """
        try:
            return float(frame[security].iloc[idx])
        except Exception:
            return 0.0

    def __call__(self, target) -> bool:
        """
        Compute hedge notionals and execute hedge trades.

        Parameters
        ----------
        target : StrategyBase
            The node whose risks are being hedged.

        Returns
        -------
        bool
            Always returns True.
        """
        if "selected" not in target.temp:
            raise ValueError("HedgeRisks requires target.temp['selected'].")

        hedge_instruments = target.temp["selected"]

        # -------------------------------------------------------------- #
        # 1. Build risk vector (target risk + optional strategy risk).
        # -------------------------------------------------------------- #
        target_risk = np.array([self._get_risk_value(target, m) for m in self.measures])

        if self.strategy is not None:
            strategy_risk = np.array(
                [self._get_risk_value(self.strategy, m) for m in self.measures]
            )
            target_risk = target_risk + strategy_risk

        # Convert to column vector: shape (n_measures, 1)
        target_risk = target_risk.reshape(-1, 1)

        # -------------------------------------------------------------- #
        # 2. Build the hedge Jacobian matrix
        #    (rows = instruments, cols = measures).
        # -------------------------------------------------------------- #
        unit_risk_data = target.get_data("unit_risk")
        current_date = target.now

        # Pre-resolve all measure frames + date locations
        measure_frames = []
        for m in self.measures:
            frame = unit_risk_data.get(m)
            if frame is None:
                raise ValueError(
                    f"Unit risk data for measure '{m}' not present on target '{target.name}'."
                )
            try:
                idx = frame.index.get_loc(current_date)
            except KeyError:
                raise ValueError(
                    f"Date {current_date} missing from unit risk data for measure '{m}'."
                )
            measure_frames.append((frame, idx))

        # Construct Jacobian: shape (n_instruments, n_measures)
        hedge_risk = np.zeros((len(hedge_instruments), len(self.measures)), dtype=float)

        for i, sec in enumerate(hedge_instruments):
            for j, (frame, idx) in enumerate(measure_frames):
                hedge_risk[i, j] = self._get_unit_risk(sec, frame, idx)

        # -------------------------------------------------------------- #
        # 3. Compute hedge notionals
        # -------------------------------------------------------------- #
        if self.pseudo:
            inv = np.linalg.pinv(hedge_risk).T
        else:
            if hedge_risk.shape[0] != hedge_risk.shape[1]:
                raise ValueError(
                    f"Jacobian must be square when pseudo=False. "
                    f"Got {hedge_risk.shape[0]} instruments and {hedge_risk.shape[1]} measures."
                )
            inv = np.linalg.inv(hedge_risk).T

        notionals = (inv @ -target_risk).flatten()

        # -------------------------------------------------------------- #
        # 4. Execute hedges
        # -------------------------------------------------------------- #
        for notional, sec in zip(notionals, hedge_instruments):
            if np.isnan(notional):
                if self.throw_nan:
                    raise ValueError(f"Hedge notional for {sec} is NaN.")
                else:
                    continue
            target.transact(notional, sec)

        return True
