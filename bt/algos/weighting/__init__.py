"""
Weighting Algos Package

This package contains various Algos to generate or adjust portfolio weights
for a strategy. Each Algo modifies `temp['weights']` in the target strategy.

Available Algos:
- WeightEqually
- WeighSpecified
- ScaleWeights
- WeighTarget
- WeightInvVol
- WeightMarket
- WeightRandomly
- WeightRiskParity
- WeightMeanVar
- WeightMinVar
- WeightMaxDiversification
- TargetVol
- LimitDeltas
- LimitWeights
"""

from .equal import WeightEqually
from .core import WeightAlgo
from .specified import WeighSpecified
from .scale import ScaleWeights
from .specified import WeighTarget
from .inv_vol import WeightInvVol
from .market import WeightMarket
from .random import WeightRandomly
from .risk_parity import WeightRiskParity
from .mean_variance import WeightMeanVar
from .min_variance import WeightMinVar
from .max_diversification import WeightMaxDiversification
from .target_vol import TargetVol
from .limits import LimitDeltas
from .limits import LimitWeights

__all__ = [
    "WeightEqually",
    "WeightAlgo",
    "WeighSpecified",
    "ScaleWeights",
    "WeighTarget",
    "WeightInvVol",
    "WeightMarket",
    "WeightRandomly",
    "WeightRiskParity",
    "WeightMeanVar",
    "WeightMinVar",
    "WeightMaxDiversification",
    "TargetVol",
    "LimitDeltas",
    "LimitWeights",
]
