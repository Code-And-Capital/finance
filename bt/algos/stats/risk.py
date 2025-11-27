from bt.core.algo_base import Algo, SecurityBase
import pandas as pd
import numpy as np
from utils.math_utils import is_zero


class UpdateRisk(Algo):
    """
    Compute and attach a risk measure to a strategy tree, recursively.

    This Algo calculates a per-node risk value based on “unit risk” inputs
    provided via the ``additional_data`` argument passed to
    :class:`bt.Backtest`. The user must provide:

    additional_data = {
        "unit_risk": {
            <measure_name>: DataFrame indexed by date, columns = securities
        }
    }

    For example:
        "IR01": DataFrame(...)
        "PVBP": DataFrame(...)
        "Beta": DataFrame(...)

    The Algo walks the entire strategy tree each period and:
        * Computes security-level risk (position × unit_risk × multiplier)
        * Aggregates child risks for composite nodes
        * Optionally records a full time series of the selected risk measure

    Parameters
    ----------
    measure : str
        Name of the risk measure to compute (must match a key under
        additional_data["unit_risk"]).
    history : int, default=0
        Depth at which historical timeseries should be recorded.
        0 = no history, 1 = top node only, 2 = top two levels, etc.

    Modifies
    --------
    target.risk[measure] : float
        Scalar risk value for this node at the current timestamp.
    target.risks[measure] : pd.Series
        Historical risk values (only if history > depth).
    """

    def __init__(self, measure: str, history: int = 0):
        super().__init__(name=f"UpdateRisk>{measure}")
        self.measure = measure
        self.history = history

    def _setup_risk(self, target, create_history: bool) -> None:
        """
        Initialize the risk containers on a node.

        Parameters
        ----------
        target : Node
            Strategy node whose risk attributes must be created.
        create_history : bool
            If True, create an empty DataFrame to store risk history.
        """
        target.risk = {}
        if create_history:
            target.risks = pd.DataFrame(index=target.data.index)

    def _setup_measure(self, target, create_history: bool) -> None:
        """
        Initialize this specific measure for a node.

        Parameters
        ----------
        target : Node
            Strategy node whose risk measure is being initialized.
        create_history : bool
            If True, create a column in the history DataFrame.
        """
        target.risk[self.measure] = np.nan
        if create_history:
            target.risks[self.measure] = np.nan

    def _get_unit_risk(self, security: str, df: pd.DataFrame, index: int) -> float:
        """
        Fetch unit risk for a single security and date.

        Parameters
        ----------
        security : str
            Security name (must correspond to a column in df).
        df : pd.DataFrame
            Unit-risk timeseries DataFrame.
        index : int
            Row index (positional) to retrieve.

        Returns
        -------
        float
            Unit risk value; returns 0.0 if missing.
        """
        try:
            return df[security].iloc[index]
        except Exception:
            return 0.0

    def _set_risk_recursive(
        self, target, depth: int, unit_risk_frame: pd.DataFrame
    ) -> None:
        """
        Recursively compute risk for the node and its children.

        Parameters
        ----------
        target : Node
            Current strategy node.
        depth : int
            Depth of this node within the strategy tree (root = 0).
        unit_risk_frame : pd.DataFrame
            Unit risk DataFrame for the selected measure.
        """
        create_history = depth < self.history

        # Initialize missing containers
        if not hasattr(target, "risk"):
            self._setup_risk(target, create_history)
        if self.measure not in target.risk:
            self._setup_measure(target, create_history)

        # Leaf-node: security
        if isinstance(target, SecurityBase):
            idx = unit_risk_frame.index.get_loc(target.root.now)
            unit_risk = self._get_unit_risk(target.name, unit_risk_frame, idx)
            risk_val = (
                0.0
                if is_zero(target.position)
                else unit_risk * target.position * target.multiplier
            )

        # Composite node: sum child risks
        else:
            risk_val = 0.0
            for child in target.children.values():
                self._set_risk_recursive(child, depth + 1, unit_risk_frame)
                risk_val += child.risk[self.measure]

        # Assign scalar
        target.risk[self.measure] = risk_val

        # Optional history
        if create_history:
            target.risks.loc[target.now, self.measure] = risk_val

    def __call__(self, target) -> bool:
        """
        Compute the risk measure for the entire tree at the current timestamp.

        Parameters
        ----------
        target : Node
            The root of the strategy tree at the current time.

        Returns
        -------
        bool
            Always True.
        """
        unit_risk_frame = target.get_data("unit_risk")[self.measure]
        self._set_risk_recursive(target, 0, unit_risk_frame)
        return True
