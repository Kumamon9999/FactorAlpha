"""Tests for factor computation and cross-sectional transforms."""

import numpy as np
import pandas as pd
import pytest

from factor_alpha.factors import (
    MomentumFactor,
    ShortTermReversalFactor,
    LowVolatilityFactor,
    IdiosyncraticVolatilityFactor,
    PrecomputedFactor,
    cross_sectional_zscore,
    rank_normalize,
    composite_score,
    neutralize,
)
from factor_alpha.factors.ml import (
    PCAResidualFactor,
    RidgeAlphaFactor,
    GradientBoostFactor,
    RandomForestFactor,
)


# ---------------------------------------------------------------------------
# MomentumFactor
# ---------------------------------------------------------------------------

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


def test_momentum_lookback_less_than_skip_raises():
    with pytest.raises(ValueError):
        MomentumFactor(lookback=10, skip=20)


def test_momentum_name():
    f = MomentumFactor(lookback=252, skip=21)
    assert "252" in f.name and "21" in f.name


# ---------------------------------------------------------------------------
# ShortTermReversalFactor
# ---------------------------------------------------------------------------

def test_reversal_anticorrelated_with_short_returns(returns, prices):
    """Reversal scores should be negatively correlated with recent returns."""
    f = ShortTermReversalFactor(lookback=5)
    scores = f.compute(prices, returns)
    recent = returns.rolling(5).sum()
    valid = scores.notna() & recent.notna()
    corr = np.corrcoef(scores.values[valid.values], recent.values[valid.values])[0, 1]
    assert corr < -0.9


def test_reversal_name():
    assert "5" in ShortTermReversalFactor(lookback=5).name


# ---------------------------------------------------------------------------
# LowVolatilityFactor
# ---------------------------------------------------------------------------

def test_low_vol_negative_of_vol(returns, prices):
    f = LowVolatilityFactor(lookback=30)
    scores = f.compute(prices, returns)
    expected_vol = returns.rolling(30).std() * np.sqrt(252)
    diff = (scores + expected_vol).dropna()
    assert diff.abs().max().max() < 1e-10


def test_low_vol_scores_non_positive(returns, prices):
    """Scores are negated volatility, so all valid values should be negative."""
    f = LowVolatilityFactor(lookback=30)
    scores = f.compute(prices, returns).dropna()
    assert (scores <= 0).all().all()


# ---------------------------------------------------------------------------
# IdiosyncraticVolatilityFactor
# ---------------------------------------------------------------------------

def test_idio_vol_shape(returns, prices):
    f = IdiosyncraticVolatilityFactor(lookback=60)
    scores = f.compute(prices, returns)
    assert scores.shape == returns.shape


def test_idio_vol_scores_non_positive(returns, prices):
    """Idio vol is negated, so all valid scores should be ≤ 0."""
    f = IdiosyncraticVolatilityFactor(lookback=60)
    scores = f.compute(prices, returns).dropna()
    assert (scores <= 0 + 1e-9).all().all()


def test_idio_vol_nan_prefix(returns, prices):
    f = IdiosyncraticVolatilityFactor(lookback=60)
    scores = f.compute(prices, returns)
    assert scores.iloc[:59].isna().all().all()


def test_idio_vol_name():
    assert "60" in IdiosyncraticVolatilityFactor(lookback=60).name


# ---------------------------------------------------------------------------
# Cross-sectional transforms
# ---------------------------------------------------------------------------

def test_zscore_zero_mean_unit_std(factor_scores):
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
    assert comp.abs().mean().mean() < 0.1


def test_composite_custom_weights(factor_scores):
    d = {"a": factor_scores, "b": factor_scores * 2}
    comp = composite_score(d, weights={"a": 1.0, "b": 0.0}, normalize=False)
    np.testing.assert_allclose(comp.values, factor_scores.values, atol=1e-9)


def test_composite_normalize_reruns_zscore(factor_scores):
    d = {"a": factor_scores, "b": factor_scores * 3}
    comp = composite_score(d, normalize=True)
    # after re-zscore+winsorize, row means are near zero (not exact due to clipping)
    row_means = comp.mean(axis=1).dropna()
    assert row_means.abs().max() < 0.1


def test_neutralize_removes_group_mean(factor_scores):
    tickers = factor_scores.columns.tolist()
    groups = pd.Series(
        ["G1"] * (len(tickers) // 2) + ["G2"] * (len(tickers) - len(tickers) // 2),
        index=tickers,
    )
    neutralized = neutralize(factor_scores, groups)
    # within each group, the mean of neutralized scores should be ~0
    for grp in ["G1", "G2"]:
        members = groups[groups == grp].index.tolist()
        cols = [c for c in factor_scores.columns if c in members]
        group_means = neutralized[cols].mean(axis=1).dropna()
        assert group_means.abs().max() < 1e-9


def test_neutralize_shape_preserved(factor_scores):
    groups = pd.Series("G1", index=factor_scores.columns)
    out = neutralize(factor_scores, groups)
    assert out.shape == factor_scores.shape


# ---------------------------------------------------------------------------
# PrecomputedFactor
# ---------------------------------------------------------------------------

def test_precomputed_factor_reindexes_to_returns(returns):
    scores = returns.iloc[10:20, :5] * 0 + 1
    factor = PrecomputedFactor("intraday_score", scores)
    out = factor.compute(returns, returns)

    assert out.shape == returns.shape
    assert out.loc[scores.index, scores.columns].eq(1).all().all()
    assert out.drop(index=scores.index).isna().all().all()


def test_precomputed_factor_empty_raises(returns):
    with pytest.raises(ValueError):
        PrecomputedFactor("bad", pd.DataFrame())


def test_precomputed_factor_name(returns):
    f = PrecomputedFactor("my_signal", returns.iloc[:5])
    assert f.name == "my_signal"


# ---------------------------------------------------------------------------
# ML Factors — shape and output sanity (small data, no look-ahead)
# ---------------------------------------------------------------------------

@pytest.fixture
def short_returns():
    """130-day returns for 10 tickers — just enough for ML factors."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-03", periods=130, freq="B")
    tickers = [f"S{i}" for i in range(10)]
    data = rng.normal(0, 0.01, (130, 10))
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def short_prices(short_returns):
    return 100 * np.exp(short_returns.cumsum())


def test_pca_residual_shape(short_returns, short_prices):
    f = PCAResidualFactor(n_components=2, signal_window=5)
    scores = f.compute(short_prices, short_returns)
    assert scores.shape == short_returns.shape


def test_pca_residual_no_all_nan(short_returns, short_prices):
    f = PCAResidualFactor(n_components=2, signal_window=5)
    scores = f.compute(short_prices, short_returns)
    assert scores.notna().any().any()


def test_ridge_alpha_shape(short_returns, short_prices):
    f = RidgeAlphaFactor(forward=1, min_obs=63)
    scores = f.compute(short_prices, short_returns)
    assert scores.shape == short_returns.shape


def test_ridge_alpha_last_row_filled(short_returns, short_prices):
    f = RidgeAlphaFactor(forward=1, min_obs=63)
    scores = f.compute(short_prices, short_returns)
    assert scores.iloc[-1].notna().any()


def test_gbm_factor_shape(short_returns, short_prices):
    f = GradientBoostFactor(forward=1, n_estimators=10, min_obs=63)
    scores = f.compute(short_prices, short_returns)
    assert scores.shape == short_returns.shape


def test_gbm_factor_last_row_filled(short_returns, short_prices):
    f = GradientBoostFactor(forward=1, n_estimators=10, min_obs=63)
    scores = f.compute(short_prices, short_returns)
    assert scores.iloc[-1].notna().any()


def test_rf_factor_shape(short_returns, short_prices):
    f = RandomForestFactor(forward=1, n_estimators=10, min_obs=63)
    scores = f.compute(short_prices, short_returns)
    assert scores.shape == short_returns.shape


def test_rf_factor_feature_importances(short_returns, short_prices):
    f = RandomForestFactor(forward=1, n_estimators=10, min_obs=63)
    f.compute(short_prices, short_returns)
    assert hasattr(f, "feature_importances_")
    assert len(f.feature_importances_) > 0


def test_rf_factor_too_short_returns_empty(short_returns, short_prices):
    """When T < min_obs + forward + 1, scores should be all NaN."""
    tiny = short_returns.iloc[:10]
    tiny_prices = short_prices.iloc[:10]
    f = RandomForestFactor(forward=1, n_estimators=5, min_obs=63)
    scores = f.compute(tiny_prices, tiny)
    assert scores.isna().all().all()
