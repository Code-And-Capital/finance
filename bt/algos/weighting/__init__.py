"""
Weighting Algos Package

This package contains various Algos to generate or adjust portfolio weights
for a strategy. Each Algo modifies `temp['weights']` in the target strategy.

Available Algos:
- WeighEqually
- WeighSpecified
- ScaleWeights
- WeighTarget
- WeighInvVol
- WeighRandomly
- WeighERC
- WeighMeanVar
- TargetVol
- LimitDeltas
- LimitWeights
"""

from .equal import WeighEqually
from .specified import WeighSpecified
from .scale import ScaleWeights
from .specified import WeighTarget
from .inv_vol import WeighInvVol
from .random import WeighRandomly
from .risk_parity import WeighERC
from .mvo import WeighMeanVar
from .target_vol import TargetVol
from .limits import LimitDeltas
from .limits import LimitWeights

__all__ = [
    "WeighEqually",
    "WeighSpecified",
    "ScaleWeights",
    "WeighTarget",
    "WeighInvVol",
    "WeighRandomly",
    "WeighERC",
    "WeighMeanVar",
    "TargetVol",
    "LimitDeltas",
    "LimitWeights",
]
