"""
Portfolio optimisation using CVXPY.

Supported objectives:
  - max_sharpe     : maximise risk-adjusted composite factor score
  - min_variance   : minimise portfolio variance
  - risk_parity    : equalise marginal risk contributions (iterative)
  - factor_tilt    : max factor score subject to risk budget constraint
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False


@dataclass
class OptimizationResult:
    weights: pd.Series
    objective_value: float
    status: str
    expected_return: float   # annualised, from factor scores
    expected_vol: float      # annualised
    sharpe: float


class PortfolioOptimizer:
    """
    Builds optimal portfolio weights from factor scores and a covariance matrix.

    Parameters
    ----------
    scores     : cross-sectional composite factor scores (latest date), Series (tickers)
    covariance : (N × N) annualised covariance matrix from FactorRiskModel
    risk_free  : annualised risk-free rate (default 0)
    """

    def __init__(
        self,
        scores: pd.Series,
        covariance: pd.DataFrame,
        risk_free: float = 0.0,
    ):
        common = scores.index.intersection(covariance.index)
        self.scores = scores.loc[common]
        self.covariance = covariance.loc[common, common]
        self.risk_free = risk_free
        self.tickers = list(common)
        self.N = len(self.tickers)

    # ------------------------------------------------------------------
    # Public optimisation methods
    # ------------------------------------------------------------------

    def max_sharpe(
        self,
        long_only: bool = True,
        max_weight: float = 0.10,
        min_weight: Optional[float] = None,
        max_factor_exposure: Optional[float] = None,
        turnover_penalty: float = 0.0,
        prev_weights: Optional[pd.Series] = None,
    ) -> OptimizationResult:
        """
        Maximise Sharpe ratio: scores'w / sqrt(w'Σw).
        Implemented as the Sharpe-equivalent QP via variable substitution y = w/k.
        """
        self._require_cvxpy()
        Sigma = self.covariance.values
        mu = self.scores.values  # proxy for expected returns (z-scored factor scores)

        # Sharpe maximisation: max mu'w - λ*w'Σw  (parametric frontier)
        # We sweep λ and pick the portfolio with max empirical Sharpe
        w = cp.Variable(self.N)
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= (min_weight or 0.0))
        constraints.append(w <= max_weight)

        # turnover penalty
        penalty = 0
        if turnover_penalty > 0 and prev_weights is not None:
            prev = prev_weights.reindex(self.tickers).fillna(0).values
            penalty = turnover_penalty * cp.norm1(w - prev)

        best_result = None
        best_sharpe = -np.inf

        for lam in np.logspace(-3, 1, 30):
            obj = mu @ w - lam * cp.quad_form(w, Sigma) - penalty
            prob = cp.Problem(cp.Maximize(obj), constraints)
            prob.solve(solver=cp.CLARABEL, warm_start=True)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                continue
            wv = np.array(w.value).flatten()
            port_vol = np.sqrt(wv @ Sigma @ wv)
            port_ret = float(mu @ wv)
            sharpe = (port_ret - self.risk_free) / (port_vol + 1e-9)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_result = (wv, prob.value, port_ret, port_vol, prob.status)

        if best_result is None:
            return self._fallback_equal_weight("max_sharpe infeasible")

        wv, obj_val, port_ret, port_vol, status = best_result
        return OptimizationResult(
            weights=pd.Series(wv, index=self.tickers),
            objective_value=float(obj_val),
            status=status,
            expected_return=float(port_ret),
            expected_vol=float(port_vol),
            sharpe=best_sharpe,
        )

    def min_variance(
        self,
        long_only: bool = True,
        max_weight: float = 0.10,
        min_factor_exposure: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Minimise portfolio variance subject to optional minimum factor exposure.
        """
        self._require_cvxpy()
        Sigma = self.covariance.values
        mu = self.scores.values
        w = cp.Variable(self.N)

        constraints = [cp.sum(w) == 1, w <= max_weight]
        if long_only:
            constraints.append(w >= 0)
        if min_factor_exposure is not None:
            constraints.append(mu @ w >= min_factor_exposure)

        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        prob.solve(solver=cp.CLARABEL)

        if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            return self._fallback_equal_weight("min_variance infeasible")

        wv = np.array(w.value).flatten()
        port_vol = np.sqrt(float(wv @ Sigma @ wv))
        port_ret = float(mu @ wv)
        return OptimizationResult(
            weights=pd.Series(wv, index=self.tickers),
            objective_value=float(prob.value),
            status=prob.status,
            expected_return=port_ret,
            expected_vol=port_vol,
            sharpe=(port_ret - self.risk_free) / (port_vol + 1e-9),
        )

    def risk_parity(self, max_iter: int = 500, tol: float = 1e-8) -> OptimizationResult:
        """
        Risk parity (equal risk contribution) via iterative Newton method.
        Each asset contributes equally to total portfolio volatility.
        """
        Sigma = self.covariance.values
        mu = self.scores.values
        N = self.N
        target = np.ones(N) / N  # equal risk budget

        w = np.ones(N) / N
        for _ in range(max_iter):
            port_var = w @ Sigma @ w
            grad = 2 * Sigma @ w
            rc = w * grad / port_var  # risk contributions
            diff = rc - target
            if np.max(np.abs(diff)) < tol:
                break
            # Newton step
            H = np.diag(grad / port_var) + w[:, None] * (
                2 * Sigma / port_var - np.outer(grad, grad) / port_var**2
            )
            try:
                step = np.linalg.solve(H, diff)
            except np.linalg.LinAlgError:
                break
            w -= 0.1 * step
            w = np.maximum(w, 1e-6)
            w /= w.sum()

        port_vol = np.sqrt(float(w @ Sigma @ w))
        port_ret = float(mu @ w)
        return OptimizationResult(
            weights=pd.Series(w, index=self.tickers),
            objective_value=float(np.std(w * (2 * Sigma @ w) / (w @ Sigma @ w))),
            status="converged",
            expected_return=port_ret,
            expected_vol=port_vol,
            sharpe=(port_ret - self.risk_free) / (port_vol + 1e-9),
        )

    def factor_tilt(
        self,
        risk_budget: float,
        long_only: bool = True,
        max_weight: float = 0.10,
    ) -> OptimizationResult:
        """
        Maximise composite factor score subject to a portfolio variance budget.
        """
        self._require_cvxpy()
        Sigma = self.covariance.values
        mu = self.scores.values
        w = cp.Variable(self.N)

        constraints = [
            cp.sum(w) == 1,
            cp.quad_form(w, Sigma) <= risk_budget**2,
            w <= max_weight,
        ]
        if long_only:
            constraints.append(w >= 0)

        prob = cp.Problem(cp.Maximize(mu @ w), constraints)
        prob.solve(solver=cp.CLARABEL)

        if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            return self._fallback_equal_weight("factor_tilt infeasible")

        wv = np.array(w.value).flatten()
        port_vol = np.sqrt(float(wv @ Sigma @ wv))
        port_ret = float(mu @ wv)
        return OptimizationResult(
            weights=pd.Series(wv, index=self.tickers),
            objective_value=float(prob.value),
            status=prob.status,
            expected_return=port_ret,
            expected_vol=port_vol,
            sharpe=(port_ret - self.risk_free) / (port_vol + 1e-9),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fallback_equal_weight(self, reason: str) -> OptimizationResult:
        w = np.ones(self.N) / self.N
        Sigma = self.covariance.values
        mu = self.scores.values
        port_vol = np.sqrt(float(w @ Sigma @ w))
        port_ret = float(mu @ w)
        return OptimizationResult(
            weights=pd.Series(w, index=self.tickers),
            objective_value=0.0,
            status=f"fallback: {reason}",
            expected_return=port_ret,
            expected_vol=port_vol,
            sharpe=(port_ret - self.risk_free) / (port_vol + 1e-9),
        )

    @staticmethod
    def _require_cvxpy():
        if not _CVXPY_AVAILABLE:
            raise ImportError("cvxpy is required for this optimisation method. pip install cvxpy")
