"""
Barra-style statistical factor risk model.

Model:  r = B @ f + ε
  r  (T × N)  asset log-returns
  B  (N × K)  factor loadings (exposures)
  f  (T × K)  factor returns
  ε  (T × N)  idiosyncratic returns

Portfolio risk:
  σ²_p = w' (B Σ_f B' + D) w
  where Σ_f = factor cov (K×K), D = diag(σ²_ε) (N×N)
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RiskAttribution:
    total_vol: float               # annualised portfolio vol
    factor_vol: float              # vol from factor exposures
    idio_vol: float                # vol from idiosyncratic risk
    factor_contributions: pd.Series  # per-factor annualised vol contribution
    var_95: float                  # parametric 95% 1-day VaR (positive number)
    var_99: float
    cvar_95: float                 # Expected Shortfall at 95%
    cvar_99: float

    def to_dict(self) -> Dict:
        return {
            "total_vol": round(self.total_vol, 6),
            "factor_vol": round(self.factor_vol, 6),
            "idio_vol": round(self.idio_vol, 6),
            "var_95": round(self.var_95, 6),
            "var_99": round(self.var_99, 6),
            "cvar_95": round(self.cvar_95, 6),
            "cvar_99": round(self.cvar_99, 6),
            "factor_contributions": self.factor_contributions.round(6).to_dict(),
        }


class FactorRiskModel:
    """
    Fits a K-factor risk model via OLS time-series regression and exposes
    portfolio risk decomposition and VaR attribution.

    Parameters
    ----------
    returns   : (T × N) asset log-returns
    factors   : dict of name → (T × N) factor score DataFrames  OR
                a single (T × K) factor returns DataFrame.
    use_scores: if True, `factors` is a dict of cross-sectional score DataFrames
                and factor *returns* are constructed as score-weighted portfolios.
                If False, `factors` is already a (T × K) factor return DataFrame.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        factors: Dict[str, pd.DataFrame],
        use_scores: bool = True,
        min_history: int = 60,
    ):
        self.asset_returns = returns.copy()
        self.min_history = min_history

        if use_scores:
            self.factor_returns = self._scores_to_factor_returns(factors, returns)
        else:
            # factors is already {name: Series/1-col DataFrame of factor returns}
            self.factor_returns = pd.DataFrame(
                {k: v.reindex(returns.index) for k, v in factors.items()}
            ).dropna()

        common_idx = self.asset_returns.index.intersection(self.factor_returns.index)
        self.asset_returns = self.asset_returns.loc[common_idx]
        self.factor_returns = self.factor_returns.loc[common_idx]

        self._B: Optional[pd.DataFrame] = None          # loadings N × K
        self._sigma_f: Optional[pd.DataFrame] = None    # factor cov K × K
        self._idio_var: Optional[pd.Series] = None      # idiosyncratic var, length N

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit(self) -> "FactorRiskModel":
        """Run OLS regression of each asset return on factor returns."""
        F = self.factor_returns.values          # T × K
        F_with_const = np.column_stack([np.ones(len(F)), F])  # T × (K+1)
        R = self.asset_returns.values           # T × N

        # OLS: B_hat = (F'F)^{-1} F'R  (intercept dropped from loadings)
        coeffs, residuals, _, _ = np.linalg.lstsq(F_with_const, R, rcond=None)
        # coeffs shape: (K+1) × N  — row 0 is intercept
        B = coeffs[1:].T  # N × K

        fitted = F @ B.T  # T × N
        resid = R - fitted - coeffs[0]  # subtract intercept

        self._B = pd.DataFrame(B, index=self.asset_returns.columns, columns=self.factor_returns.columns)
        self._sigma_f = pd.DataFrame(
            np.cov(F.T) if F.shape[1] > 1 else np.atleast_2d(np.var(F)),
            index=self.factor_returns.columns,
            columns=self.factor_returns.columns,
        )
        self._idio_var = pd.Series(
            np.var(resid, axis=0), index=self.asset_returns.columns
        )
        return self

    # ------------------------------------------------------------------
    # Risk decomposition
    # ------------------------------------------------------------------

    def total_covariance(self) -> pd.DataFrame:
        """Full N × N covariance matrix = B Σ_f B' + D."""
        self._check_fitted()
        B = self._B.values
        factor_cov = B @ self._sigma_f.values @ B.T
        idio_cov = np.diag(self._idio_var.values)
        total = factor_cov + idio_cov
        return pd.DataFrame(total, index=self._B.index, columns=self._B.index)

    def portfolio_variance(self, weights: pd.Series) -> float:
        """Annualised portfolio variance given asset weights."""
        self._check_fitted()
        w = weights.reindex(self._B.index).fillna(0).values
        Sigma = self.total_covariance().values
        return float(w @ Sigma @ w) * 252

    def attribute(self, weights: pd.Series) -> RiskAttribution:
        """Full risk attribution for a portfolio weight vector."""
        self._check_fitted()
        w = weights.reindex(self._B.index).fillna(0).values
        B = self._B.values                     # N × K
        Sigma_f = self._sigma_f.values         # K × K
        D = np.diag(self._idio_var.values)     # N × N

        factor_var_ann = float(w @ B @ Sigma_f @ B.T @ w) * 252
        idio_var_ann = float(w @ D @ w) * 252
        total_var_ann = factor_var_ann + idio_var_ann

        total_vol = np.sqrt(max(total_var_ann, 0))
        factor_vol = np.sqrt(max(factor_var_ann, 0))
        idio_vol = np.sqrt(max(idio_var_ann, 0))

        # per-factor contribution: marginal variance = 2 * w'B_k * (Σ_f B'w)_k
        exposure = B.T @ w                     # K-vector: factor exposures
        factor_var_contrib = Sigma_f @ exposure * exposure * 252
        factor_contributions = pd.Series(
            np.sqrt(np.maximum(factor_var_contrib, 0)),
            index=self._B.columns,
        )

        # parametric VaR and CVaR (normal assumption)
        daily_vol = total_vol / np.sqrt(252)
        var_95 = float(stats.norm.ppf(0.95) * daily_vol)
        var_99 = float(stats.norm.ppf(0.99) * daily_vol)
        cvar_95 = float(stats.norm.pdf(stats.norm.ppf(0.95)) / 0.05 * daily_vol)
        cvar_99 = float(stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01 * daily_vol)

        return RiskAttribution(
            total_vol=total_vol,
            factor_vol=factor_vol,
            idio_vol=idio_vol,
            factor_contributions=factor_contributions,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
        )

    def historical_var(
        self, weights: pd.Series, confidence: float = 0.95
    ) -> float:
        """Historical-simulation VaR using the fitted asset returns."""
        self._check_fitted()
        w = weights.reindex(self.asset_returns.columns).fillna(0)
        port_ret = self.asset_returns @ w
        return float(-np.percentile(port_ret, (1 - confidence) * 100))

    # ------------------------------------------------------------------
    # Factor return construction
    # ------------------------------------------------------------------

    @staticmethod
    def _scores_to_factor_returns(
        factor_scores: Dict[str, pd.DataFrame], returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build factor return series from cross-sectional score DataFrames.
        Factor return on day t = score-weighted average of asset returns.
        Scores are rank-normalised row-wise before weighting.
        """
        factor_rets = {}
        for name, scores in factor_scores.items():
            common_idx = scores.index.intersection(returns.index)
            common_cols = scores.columns.intersection(returns.columns)
            s = scores.loc[common_idx, common_cols]
            r = returns.loc[common_idx, common_cols]

            # rank-normalise each row to sum to zero (long-short)
            ranks = s.rank(axis=1)
            ranks = ranks.subtract(ranks.mean(axis=1), axis=0)
            norms = ranks.abs().sum(axis=1).replace(0, np.nan)
            w = ranks.divide(norms, axis=0)

            factor_ret = (w * r).sum(axis=1)
            factor_rets[name] = factor_ret

        return pd.DataFrame(factor_rets).dropna()

    def _check_fitted(self):
        if self._B is None:
            raise RuntimeError("Call .fit() before accessing risk attributes.")
