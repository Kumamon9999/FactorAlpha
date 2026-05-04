"""Tests for FactorResearch (IC, decay, quintile, turnover)."""

import numpy as np
import pandas as pd
import pytest

from factor_alpha.analysis.factor_research import FactorResearch


def test_ic_range(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    ic = fr.ic(forward_period=1).dropna()
    assert len(ic) > 0, "IC series should not be empty"
    assert ic.between(-1, 1).all(), "IC values must be in [-1, 1]"


def test_ic_has_signal_for_constructed_factor(returns):
    """A factor equal to next-day returns should have IC ≈ 1."""
    fwd = returns.shift(-1).fillna(0)
    fr = FactorResearch(fwd, returns)
    ic = fr.ic(forward_period=1)
    assert ic.mean() > 0.5


def test_ic_pearson_in_range(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    ic = fr.ic(forward_period=1, method="pearson").dropna()
    assert ic.between(-1, 1).all()


def test_ic_invalid_method_raises(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    with pytest.raises(Exception):
        fr.ic(forward_period=1, method="kendall").dropna()


def test_icir_positive_for_good_factor(returns):
    fwd = returns.shift(-1).fillna(0)
    fr = FactorResearch(fwd, returns)
    assert fr.icir(1) > 0


def test_icir_negative_for_anti_factor(returns):
    anti = -returns.shift(-1).fillna(0)
    fr = FactorResearch(anti, returns)
    assert fr.icir(1) < 0


def test_decay_profile_length(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    decay = fr.decay_profile(max_horizon=10)
    assert len(decay) == 10


def test_decay_profile_index_is_horizons(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    decay = fr.decay_profile(max_horizon=5)
    assert list(decay.index) == [1, 2, 3, 4, 5]


def test_quintile_returns_shape(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    qr = fr.quintile_returns(forward_periods=[1, 5], n_quintiles=5)
    assert qr.shape == (2, 5)


def test_quintile_returns_columns_labeled(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    qr = fr.quintile_returns(forward_periods=[1], n_quintiles=5)
    assert list(qr.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_quintile_q5_gt_q1_for_good_factor(returns):
    """For a factor = next-period return, Q5 (high score) should have higher returns than Q1."""
    fwd = returns.shift(-1).fillna(0)
    fr = FactorResearch(fwd, returns)
    qr = fr.quintile_returns(forward_periods=[1], n_quintiles=5)
    assert qr.loc[1, "Q5"] > qr.loc[1, "Q1"]


def test_turnover_between_0_and_1(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    t = fr.turnover(top_pct=0.2)
    assert (t >= 0).all() and (t <= 1).all()


def test_turnover_non_empty(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    t = fr.turnover(top_pct=0.2)
    assert len(t) > 0


def test_evaluate_returns_factor_stats(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    stats = fr.evaluate(name="test_factor")
    assert stats.name == "test_factor"
    assert "mean_ic" in stats.summary
    assert "icir" in stats.summary
    assert "t_stat" in stats.summary


def test_evaluate_t_stat_finite(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    stats = fr.evaluate()
    assert np.isfinite(stats.summary["t_stat"])


def test_evaluate_mean_turnover_in_summary(factor_scores, returns):
    fr = FactorResearch(factor_scores, returns)
    stats = fr.evaluate()
    assert "mean_turnover" in stats.summary
    assert 0 <= stats.summary["mean_turnover"] <= 1
