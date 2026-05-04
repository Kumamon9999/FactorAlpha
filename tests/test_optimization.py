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
    result = PortfolioOptimizer(scores, cov).max_sharpe()
    assert abs(result.weights.sum() - 1.0) < 1e-4


def test_max_sharpe_non_negative_long_only(opt_inputs):
    scores, cov = opt_inputs
    result = PortfolioOptimizer(scores, cov).max_sharpe(long_only=True)
    assert (result.weights >= -1e-6).all()


def test_max_sharpe_respects_max_weight(opt_inputs):
    scores, cov = opt_inputs
    result = PortfolioOptimizer(scores, cov).max_sharpe(max_weight=0.05)
    assert result.weights.max() <= 0.05 + 1e-4


def test_max_sharpe_with_turnover_penalty(opt_inputs):
    scores, cov = opt_inputs
    prev = pd.Series(1.0 / len(scores), index=scores.index)
    result = PortfolioOptimizer(scores, cov).max_sharpe(turnover_penalty=0.01, prev_weights=prev)
    assert abs(result.weights.sum() - 1.0) < 1e-4


def test_min_variance_weights_sum_to_one(opt_inputs):
    scores, cov = opt_inputs
    result = PortfolioOptimizer(scores, cov).min_variance()
    assert abs(result.weights.sum() - 1.0) < 1e-4


def test_min_variance_lower_vol_than_equal_weight(opt_inputs):
    """Min-variance portfolio should have lower vol than equal-weight."""
    scores, cov = opt_inputs
    result = PortfolioOptimizer(scores, cov).min_variance()
    Sigma = cov.values
    w_opt = result.weights.values
    w_eq = np.ones(len(w_opt)) / len(w_opt)
    vol_opt = np.sqrt(w_opt @ Sigma @ w_opt)
    vol_eq = np.sqrt(w_eq @ Sigma @ w_eq)
    assert vol_opt <= vol_eq + 1e-6


def test_risk_parity_weights_sum_to_one(opt_inputs):
    scores, cov = opt_inputs
    result = PortfolioOptimizer(scores, cov).risk_parity()
    assert abs(result.weights.sum() - 1.0) < 1e-4


def test_risk_parity_non_negative(opt_inputs):
    scores, cov = opt_inputs
    result = PortfolioOptimizer(scores, cov).risk_parity()
    assert (result.weights >= -1e-6).all()


def test_risk_parity_equal_risk_contributions(opt_inputs):
    """Each asset's risk contribution should be approximately equal."""
    scores, cov = opt_inputs
    result = PortfolioOptimizer(scores, cov).risk_parity()
    w = result.weights.values
    Sigma = cov.loc[result.weights.index, result.weights.index].values
    port_var = w @ Sigma @ w
    rc = w * (2 * Sigma @ w) / port_var  # relative risk contributions
    assert rc.std() < 0.05  # contributions should be roughly equal


def test_factor_tilt_weights_sum_to_one(opt_inputs):
    scores, cov = opt_inputs
    result = PortfolioOptimizer(scores, cov).factor_tilt(risk_budget=0.20)
    assert abs(result.weights.sum() - 1.0) < 1e-4


def test_factor_tilt_respects_risk_budget(opt_inputs):
    """Portfolio vol should not exceed the risk budget."""
    scores, cov = opt_inputs
    risk_budget = 0.20
    result = PortfolioOptimizer(scores, cov).factor_tilt(risk_budget=risk_budget)
    Sigma = cov.loc[result.weights.index, result.weights.index].values
    w = result.weights.values
    port_vol = np.sqrt(w @ Sigma @ w)
    assert port_vol <= risk_budget + 1e-4


def test_result_has_sharpe(opt_inputs):
    scores, cov = opt_inputs
    result = PortfolioOptimizer(scores, cov).max_sharpe()
    assert isinstance(result.sharpe, float)


def test_result_status_string(opt_inputs):
    scores, cov = opt_inputs
    result = PortfolioOptimizer(scores, cov).min_variance()
    assert isinstance(result.status, str)
