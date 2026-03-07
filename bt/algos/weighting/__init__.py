"""
Weighting Algos Package

This package contains various Algos to generate or adjust portfolio weights
for a strategy. Each Algo modifies `temp['weights']` in the target strategy.

Available Algos:
- WeightEqually
- WeightFixed
- ScaleWeights
- WeightFixedSchedule
- WeightInvVol
- WeightMarket
- WeightRandomly
- WeightRiskParity
- WeightMeanVar
- WeightMinVar
- WeightMaxDiversification
- LimitDeltas
- LimitWeights
"""

from .equal import WeightEqually
from .core import WeightAlgo
from .fixed import WeightFixed
from .fixed import WeightFixedSchedule
from .inv_vol import WeightInvVol
from .market import WeightMarket
from .random import WeightRandomly
from .risk_parity import WeightRiskParity
from .mean_variance import WeightMeanVar
from .min_variance import WeightMinVar
from .max_diversification import WeightMaxDiversification
from .modifiers import ScaleWeights
from .modifiers import LimitDeltas
from .modifiers import LimitWeights

__all__ = [
    "WeightEqually",
    "WeightAlgo",
    "WeightFixed",
    "ScaleWeights",
    "WeightFixedSchedule",
    "WeightInvVol",
    "WeightMarket",
    "WeightRandomly",
    "WeightRiskParity",
    "WeightMeanVar",
    "WeightMinVar",
    "WeightMaxDiversification",
    "LimitDeltas",
    "LimitWeights",
]
