from .base import BaseFactor
from .cross_section import cross_sectional_zscore, rank_normalize, composite_score, neutralize
from .precomputed import PrecomputedFactor

# traditional statistical factors
from .traditional import (
    MomentumFactor,
    ShortTermReversalFactor,
    LowVolatilityFactor,
    IdiosyncraticVolatilityFactor,
)

# machine learning factors
from .ml import (
    PCAResidualFactor,
    RidgeAlphaFactor,
    GradientBoostFactor,
    RandomForestFactor,
)

# qlib-integrated factors
from .qlib import (
    build_alpha158_features,
    Alpha158Factor,
    QlibLGBFactor,
)

__all__ = [
    # shared
    "BaseFactor",
    "PrecomputedFactor",
    "cross_sectional_zscore",
    "rank_normalize",
    "composite_score",
    "neutralize",
    # traditional
    "MomentumFactor",
    "ShortTermReversalFactor",
    "LowVolatilityFactor",
    "IdiosyncraticVolatilityFactor",
    # ml
    "PCAResidualFactor",
    "RidgeAlphaFactor",
    "GradientBoostFactor",
    "RandomForestFactor",
    # qlib
    "build_alpha158_features",
    "Alpha158Factor",
    "QlibLGBFactor",
]
