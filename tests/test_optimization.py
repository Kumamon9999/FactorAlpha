"""Tests for PortfolioOptimizer."""

import numpy as np
import pandas as pd
import pytest

from factor_alpha.optimization.portfolio import PortfolioOptimizer


@pytest.fixture
def opt_inputs(factor_scores, returns):
    from factor_alpha.risk.factor_risk import FactorRiskModel
    model = FactorRiskModel(returns, {"f": factor_scores}).fit()
    cov = model.total_covariance()
    scores = factor_scores.iloc[-1].dropna()
    return scores, cov


def test_max_sharpe_weights_sum_to_one(opt_inputs):
    scores, cov = opt_inputs
    opt = PortfolioOptimizer(scores, cov)
    result = opt.max_sharpe()
    assert abs(result.weights.sum() - 1.0) < 1e-4


def test_max_sharpe_non_negative_long_only(opt_inputs):
    scores, cov = opt_inputs
    opt = PortfolioOptimizer(scores, cov)
    result = opt.max_sharpe(long_only=True)
    assert (result.weights >= -1e-6).all()


def test_max_sharpe_respects_max_weight(opt_inputs):
    scores, cov = opt_inputs
    opt = PortfolioOptimizer(scores, cov)
    result = opt.max_sharpe(max_weight=0.05)
    assert result.weights.max() <= 0.05 + 1e-4


def test_min_variance_weights_sum_to_one(opt_inputs):
    scores, cov = opt_inputs
    opt = PortfolioOptimizer(scores, cov)
    result = opt.min_variance()
    assert abs(result.weights.sum() - 1.0) < 1e-4


def test_risk_parity_weights_sum_to_one(opt_inputs):
    scores, cov = opt_inputs
    opt = PortfolioOptimizer(scores, cov)
    result = opt.risk_parity()
    assert abs(result.weights.sum() - 1.0) < 1e-4


def test_risk_parity_non_negative(opt_inputs):
    scores, cov = opt_inputs
    opt = PortfolioOptimizer(scores, cov)
    result = opt.risk_parity()
    assert (result.weights >= -1e-6).all()


def test_factor_tilt_weights_sum_to_one(opt_inputs):
    scores, cov = opt_inputs
    opt = PortfolioOptimizer(scores, cov)
    result = opt.factor_tilt(risk_budget=0.20)
    assert abs(result.weights.sum() - 1.0) < 1e-4


def test_result_has_sharpe(opt_inputs):
    scores, cov = opt_inputs
    opt = PortfolioOptimizer(scores, cov)
    result = opt.max_sharpe()
    assert isinstance(result.sharpe, float)
