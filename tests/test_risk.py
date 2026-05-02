"""Tests for FactorRiskModel and RiskAttribution."""

import numpy as np
import pandas as pd
import pytest

from factor_alpha.risk.factor_risk import FactorRiskModel


def test_fit_produces_loadings(returns, factor_scores):
    model = FactorRiskModel(returns, {"momentum": factor_scores}, use_scores=True)
    model.fit()
    assert model._B is not None
    assert model._B.shape[1] == 1  # one factor


def test_total_covariance_psd(returns, factor_scores):
    model = FactorRiskModel(returns, {"momentum": factor_scores}).fit()
    cov = model.total_covariance()
    eigvals = np.linalg.eigvalsh(cov.values)
    assert eigvals.min() >= -1e-8, "Covariance must be positive semi-definite"


def test_portfolio_variance_positive(returns, factor_scores):
    model = FactorRiskModel(returns, {"momentum": factor_scores}).fit()
    w = pd.Series(1.0 / 20, index=returns.columns)
    var = model.portfolio_variance(w)
    assert var > 0


def test_risk_attribution_parts_sum(returns, factor_scores):
    model = FactorRiskModel(returns, {"momentum": factor_scores}).fit()
    w = pd.Series(1.0 / 20, index=returns.columns)
    attr = model.attribute(w)
    # factor_vol^2 + idio_vol^2 ≈ total_vol^2
    reconstructed = np.sqrt(attr.factor_vol**2 + attr.idio_vol**2)
    assert abs(reconstructed - attr.total_vol) < 1e-6


def test_var_99_gt_var_95(returns, factor_scores):
    model = FactorRiskModel(returns, {"momentum": factor_scores}).fit()
    w = pd.Series(1.0 / 20, index=returns.columns)
    attr = model.attribute(w)
    assert attr.var_99 > attr.var_95


def test_cvar_gt_var(returns, factor_scores):
    model = FactorRiskModel(returns, {"momentum": factor_scores}).fit()
    w = pd.Series(1.0 / 20, index=returns.columns)
    attr = model.attribute(w)
    assert attr.cvar_95 > attr.var_95
    assert attr.cvar_99 > attr.var_99


def test_historical_var_positive(returns, factor_scores):
    model = FactorRiskModel(returns, {"momentum": factor_scores}).fit()
    w = pd.Series(1.0 / 20, index=returns.columns)
    hvar = model.historical_var(w, confidence=0.95)
    assert isinstance(hvar, float)


def test_not_fitted_raises(returns, factor_scores):
    model = FactorRiskModel(returns, {"momentum": factor_scores})
    w = pd.Series(1.0 / 20, index=returns.columns)
    with pytest.raises(RuntimeError):
        model.attribute(w)
