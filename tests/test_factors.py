"""Tests for factor computation and cross-sectional transforms."""

import numpy as np
import pandas as pd
import pytest

from factor_alpha.factors import (
    MomentumFactor, ShortTermReversalFactor,
    LowVolatilityFactor,
    PrecomputedFactor,
    cross_sectional_zscore, rank_normalize, composite_score,
)


def test_momentum_shape(returns, prices):
    f = MomentumFactor(lookback=60, skip=5)
    scores = f.compute(prices, returns)
    assert scores.shape == returns.shape


def test_momentum_nan_prefix(returns, prices):
    """First lookback rows should be NaN."""
    f = MomentumFactor(lookback=60, skip=5)
    scores = f.compute(prices, returns)
    assert scores.iloc[:59].isna().all().all()
    assert scores.iloc[60:].notna().any().any()


def test_reversal_anticorrelated_with_short_returns(returns, prices):
    """Reversal scores should be negatively correlated with recent returns."""
    f = ShortTermReversalFactor(lookback=5)
    scores = f.compute(prices, returns)
    recent = returns.rolling(5).sum()
    valid = scores.notna() & recent.notna()
    flat_scores = scores.values[valid.values]
    flat_recent = recent.values[valid.values]
    corr = np.corrcoef(flat_scores, flat_recent)[0, 1]
    assert corr < -0.9  # should be nearly -1 by construction


def test_low_vol_negative_of_vol(returns, prices):
    """LowVolatilityFactor should equal -annualised vol."""
    f = LowVolatilityFactor(lookback=30)
    scores = f.compute(prices, returns)
    expected_vol = returns.rolling(30).std() * np.sqrt(252)
    diff = (scores + expected_vol).dropna()
    assert diff.abs().max().max() < 1e-10


def test_momentum_lookback_less_than_skip_raises():
    with pytest.raises(ValueError):
        MomentumFactor(lookback=10, skip=20)


def test_zscore_zero_mean_unit_std(factor_scores):
    # Use winsorize_std=10 so clipping does not shift the mean
    z = cross_sectional_zscore(factor_scores, winsorize_std=10.0)
    row_means = z.mean(axis=1).dropna()
    assert row_means.abs().max() < 1e-9


def test_zscore_winsorized(factor_scores):
    z = cross_sectional_zscore(factor_scores, winsorize_std=2.0)
    assert z.max().max() <= 2.0 + 1e-9
    assert z.min().min() >= -2.0 - 1e-9


def test_rank_normalize_range(factor_scores):
    r = rank_normalize(factor_scores)
    assert r.min().min() >= -1.0 - 1e-9
    assert r.max().max() <= 1.0 + 1e-9


def test_composite_equal_weights(factor_scores):
    d = {"f1": factor_scores, "f2": -factor_scores}
    comp = composite_score(d, normalize=False)
    # equal-weighted sum of f and -f ≈ 0 before re-normalisation
    assert comp.abs().mean().mean() < 0.1


def test_composite_custom_weights(factor_scores):
    d = {"a": factor_scores, "b": factor_scores * 2}
    comp = composite_score(d, weights={"a": 1.0, "b": 0.0}, normalize=False)
    np.testing.assert_allclose(comp.values, factor_scores.values, atol=1e-9)


def test_precomputed_factor_reindexes_to_returns(returns):
    scores = returns.iloc[10:20, :5] * 0 + 1
    factor = PrecomputedFactor("intraday_score", scores)
    out = factor.compute(returns, returns)

    assert out.shape == returns.shape
    assert out.loc[scores.index, scores.columns].eq(1).all().all()
    assert out.drop(index=scores.index).isna().all().all()
