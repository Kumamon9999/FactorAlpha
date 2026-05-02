"""Cross-sectional transformation utilities applied to factor DataFrames."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def cross_sectional_zscore(factor: pd.DataFrame, winsorize_std: float = 3.0) -> pd.DataFrame:
    """
    Standardise each row (date) to zero mean and unit variance.
    Winsorises at ±winsorize_std standard deviations before scaling.
    """
    mu = factor.mean(axis=1)
    sigma = factor.std(axis=1)
    z = factor.subtract(mu, axis=0).divide(sigma.replace(0, np.nan), axis=0)
    return z.clip(-winsorize_std, winsorize_std)


def rank_normalize(factor: pd.DataFrame) -> pd.DataFrame:
    """
    Replace scores with cross-sectional ranks, then linearly scale to [-1, 1].
    More robust to outliers than z-scoring.
    """
    ranks = factor.rank(axis=1, pct=True)  # [0, 1]
    return ranks * 2 - 1  # [-1, 1]


def composite_score(
    factors: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Combine multiple factor DataFrames into a single composite signal.

    Args:
        factors:   name → factor DataFrame (already z-scored / rank-normalised)
        weights:   name → weight; defaults to equal weight
        normalize: re-zscore the composite after combining

    Returns:
        Composite factor DataFrame.
    """
    names = list(factors.keys())
    if weights is None:
        weights = {n: 1.0 / len(names) for n in names}

    total_w = sum(weights[n] for n in names)
    composite = sum(factors[n] * (weights[n] / total_w) for n in names)

    if normalize:
        composite = cross_sectional_zscore(composite)
    return composite


def neutralize(factor: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """
    Sector/industry neutralization: demean within each group each day.
    groups: Series mapping ticker → group label.
    """
    result = factor.copy()
    for group in groups.unique():
        members = groups[groups == group].index.tolist()
        cols = [c for c in factor.columns if c in members]
        if not cols:
            continue
        group_mean = factor[cols].mean(axis=1)
        result[cols] = factor[cols].subtract(group_mean, axis=0)
    return result
