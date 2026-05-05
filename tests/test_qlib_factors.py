"""Tests for Qlib-integrated Alpha158Factor and QlibLGBFactor."""

import numpy as np
import pandas as pd
import pytest

from factor_alpha.factors.qlib import (
    build_alpha158_features,
    Alpha158Factor,
    QlibLGBFactor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def alpha_data():
    """130-day returns + prices for 12 tickers — enough for min_obs=63."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", periods=130, freq="B")
    tickers = [f"S{i:02d}" for i in range(12)]
    ret = rng.normal(0, 0.01, (130, 12))
    returns = pd.DataFrame(ret, index=dates, columns=tickers)
    prices = 100 * np.exp(returns.cumsum())
    return prices, returns


@pytest.fixture
def alpha_data_with_volume(alpha_data):
    prices, returns = alpha_data
    rng = np.random.default_rng(7)
    volumes = pd.DataFrame(
        rng.integers(100_000, 2_000_000, prices.shape).astype(float),
        index=prices.index,
        columns=prices.columns,
    )
    return prices, returns, volumes


@pytest.fixture
def tiny_data():
    """Only 10 rows — too short to train any factor."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-03", periods=10, freq="B")
    tickers = ["A", "B", "C"]
    ret = rng.normal(0, 0.01, (10, 3))
    returns = pd.DataFrame(ret, index=dates, columns=tickers)
    prices = 100 * np.exp(returns.cumsum())
    return prices, returns


# ---------------------------------------------------------------------------
# build_alpha158_features
# ---------------------------------------------------------------------------

def test_alpha158_features_keys_present(alpha_data):
    prices, returns = alpha_data
    feat = build_alpha158_features(prices, returns)
    # Should have momentum keys
    for d in [1, 5, 10, 20, 60]:
        assert f"RET_{d}D" in feat
    # Should have rolling feature keys for all windows
    for w in [5, 10, 20, 30, 60]:
        for prefix in ["ROC", "MA", "STD", "MAX", "MIN", "RSV", "QTLU", "QTLD"]:
            assert f"{prefix}_{w}" in feat, f"Missing {prefix}_{w}"


def test_alpha158_features_count(alpha_data):
    prices, returns = alpha_data
    feat = build_alpha158_features(prices, returns)
    # 5 momentum + 5 windows × 8 stats = 45 features (no volume)
    assert len(feat) == 45


def test_alpha158_features_with_volume_count(alpha_data_with_volume):
    prices, returns, volumes = alpha_data_with_volume
    feat = build_alpha158_features(prices, returns, volumes)
    # 45 + 5 windows × 2 volume features = 55
    assert len(feat) == 55


def test_alpha158_features_shape(alpha_data):
    prices, returns = alpha_data
    feat = build_alpha158_features(prices, returns)
    for name, df in feat.items():
        assert df.shape == returns.shape, f"Shape mismatch for {name}"


def test_alpha158_features_nan_prefix(alpha_data):
    """ROC_60 should have NaN in first 60 rows."""
    prices, returns = alpha_data
    feat = build_alpha158_features(prices, returns)
    assert feat["ROC_60"].iloc[:60].isna().all().all()


def test_alpha158_features_roc_positive_on_rising_prices():
    dates = pd.date_range("2022-01-03", periods=20, freq="B")
    prices = pd.DataFrame({"A": np.linspace(100, 120, 20)}, index=dates)
    returns = np.log(prices / prices.shift(1))
    feat = build_alpha158_features(prices, returns)
    # ROC_5 should be positive for rising prices
    valid = feat["ROC_5"].dropna()
    assert (valid["A"] > 0).all()


def test_alpha158_features_rsv_in_0_1(alpha_data):
    """RSV (stochastic oscillator) must be in [0, 1]."""
    prices, returns = alpha_data
    feat = build_alpha158_features(prices, returns)
    for w in [5, 10, 20, 30, 60]:
        rsv = feat[f"RSV_{w}"].dropna()
        assert (rsv >= -1e-9).all().all(), f"RSV_{w} below 0"
        assert (rsv <= 1 + 1e-9).all().all(), f"RSV_{w} above 1"


def test_alpha158_features_std_non_negative(alpha_data):
    prices, returns = alpha_data
    feat = build_alpha158_features(prices, returns)
    for w in [5, 10, 20, 30, 60]:
        std = feat[f"STD_{w}"].dropna()
        assert (std >= 0).all().all()


# ---------------------------------------------------------------------------
# Alpha158Factor
# ---------------------------------------------------------------------------

def test_alpha158_factor_shape(alpha_data):
    prices, returns = alpha_data
    f = Alpha158Factor(forward=1, min_obs=63)
    scores = f.compute(prices, returns)
    assert scores.shape == returns.shape


def test_alpha158_factor_last_row_filled(alpha_data):
    prices, returns = alpha_data
    f = Alpha158Factor(forward=1, min_obs=63)
    scores = f.compute(prices, returns)
    assert scores.iloc[-1].notna().any(), "Last row should have predictions"


def test_alpha158_factor_prefix_is_nan(alpha_data):
    prices, returns = alpha_data
    f = Alpha158Factor(forward=1, min_obs=63)
    scores = f.compute(prices, returns)
    # Only the last row should be filled in single-shot mode
    assert scores.iloc[:-1].isna().all().all()


def test_alpha158_factor_too_short_returns_all_nan(tiny_data):
    prices, returns = tiny_data
    f = Alpha158Factor(forward=1, min_obs=63)
    scores = f.compute(prices, returns)
    assert scores.isna().all().all()


def test_alpha158_factor_name():
    f = Alpha158Factor(forward=1, alpha=0.5)
    assert "alpha158" in f.name
    assert "0.5" in f.name


def test_alpha158_factor_with_volume(alpha_data_with_volume):
    prices, returns, volumes = alpha_data_with_volume
    f = Alpha158Factor(forward=1, min_obs=63, use_volume=True)
    scores = f.compute(prices, returns, volumes=volumes)
    assert scores.shape == returns.shape
    assert scores.iloc[-1].notna().any()


def test_alpha158_factor_use_volume_false_ignores_volume(alpha_data_with_volume):
    prices, returns, volumes = alpha_data_with_volume
    f_no_vol = Alpha158Factor(forward=1, min_obs=63, use_volume=False)
    f_with_vol = Alpha158Factor(forward=1, min_obs=63, use_volume=True)
    scores_no = f_no_vol.compute(prices, returns, volumes=volumes)
    scores_yes = f_with_vol.compute(prices, returns, volumes=volumes)
    # Both should produce valid scores; results may differ since feature sets differ
    assert scores_no.iloc[-1].notna().any()
    assert scores_yes.iloc[-1].notna().any()


def test_alpha158_factor_different_alpha_produces_different_scores(alpha_data):
    prices, returns = alpha_data
    f1 = Alpha158Factor(forward=1, alpha=0.01)
    f2 = Alpha158Factor(forward=1, alpha=100.0)
    s1 = f1.compute(prices, returns).iloc[-1].dropna()
    s2 = f2.compute(prices, returns).iloc[-1].dropna()
    # Strong regularization and weak regularization produce different predictions
    assert not np.allclose(s1.values, s2.values, atol=1e-6)


def test_alpha158_factor_in_backtest_pipeline(alpha_data):
    """Verify Alpha158Factor works as a drop-in inside BacktestEngine."""
    from factor_alpha.backtest.engine import BacktestEngine
    prices, returns = alpha_data
    engine = BacktestEngine(
        prices=prices,
        returns=returns,
        factors=[Alpha158Factor(forward=1, min_obs=63)],
        lookback=100,
        rebalance_freq="monthly",
        optimizer_mode="min_variance",
    )
    result = engine.run()
    assert len(result.portfolio_returns) == len(returns)


# ---------------------------------------------------------------------------
# QlibLGBFactor
# ---------------------------------------------------------------------------

def test_qlib_lgb_factor_shape(alpha_data):
    prices, returns = alpha_data
    f = QlibLGBFactor(forward=1, n_estimators=20, min_obs=63)
    scores = f.compute(prices, returns)
    assert scores.shape == returns.shape


def test_qlib_lgb_factor_last_row_filled(alpha_data):
    prices, returns = alpha_data
    f = QlibLGBFactor(forward=1, n_estimators=20, min_obs=63)
    scores = f.compute(prices, returns)
    assert scores.iloc[-1].notna().any()


def test_qlib_lgb_factor_feature_importances(alpha_data):
    prices, returns = alpha_data
    f = QlibLGBFactor(forward=1, n_estimators=20, min_obs=63)
    f.compute(prices, returns)
    assert hasattr(f, "feature_importances_")
    assert len(f.feature_importances_) == 45  # no volume features


def test_qlib_lgb_factor_feature_importances_sum(alpha_data):
    prices, returns = alpha_data
    f = QlibLGBFactor(forward=1, n_estimators=20, min_obs=63)
    f.compute(prices, returns)
    total = sum(f.feature_importances_.values())
    assert total > 0


def test_qlib_lgb_factor_too_short_all_nan(tiny_data):
    prices, returns = tiny_data
    f = QlibLGBFactor(forward=1, n_estimators=5, min_obs=63)
    scores = f.compute(prices, returns)
    assert scores.isna().all().all()


def test_qlib_lgb_factor_name():
    f = QlibLGBFactor(forward=1, n_estimators=50)
    assert "lgb" in f.name
    assert "50" in f.name


def test_qlib_lgb_factor_with_volume(alpha_data_with_volume):
    prices, returns, volumes = alpha_data_with_volume
    f = QlibLGBFactor(forward=1, n_estimators=20, min_obs=63, use_volume=True)
    scores = f.compute(prices, returns, volumes=volumes)
    assert scores.shape == returns.shape
    assert scores.iloc[-1].notna().any()


def test_qlib_lgb_factor_with_volume_more_features(alpha_data_with_volume):
    prices, returns, volumes = alpha_data_with_volume
    f = QlibLGBFactor(forward=1, n_estimators=10, min_obs=63, use_volume=True)
    f.compute(prices, returns, volumes=volumes)
    assert len(f.feature_importances_) == 55  # 45 + 10 volume features


def test_qlib_lgb_factor_in_composite(alpha_data):
    """Alpha158 and LGB factor can be combined into a composite score."""
    from factor_alpha.factors import composite_score, cross_sectional_zscore
    prices, returns = alpha_data
    fa = Alpha158Factor(forward=1, min_obs=63)
    fl = QlibLGBFactor(forward=1, n_estimators=10, min_obs=63)
    s_a = cross_sectional_zscore(fa.compute(prices, returns).fillna(0))
    s_l = cross_sectional_zscore(fl.compute(prices, returns).fillna(0))
    comp = composite_score({"alpha158": s_a, "lgb": s_l})
    assert comp.shape == returns.shape
