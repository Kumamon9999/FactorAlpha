"""
End-to-end factor backtest engine.

At each rebalance date:
  1. Compute factor scores on trailing window
  2. Cross-sectionally z-score and form composite
  3. Fit factor risk model on trailing window
  4. Optimise portfolio weights
  5. Record forward returns, turnover, and risk attribution
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..factors.base import BaseFactor
from ..factors.cross_section import cross_sectional_zscore, composite_score
from ..risk.factor_risk import FactorRiskModel
from ..optimization.portfolio import PortfolioOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    portfolio_returns: pd.Series          # daily portfolio log-returns
    weights_history: pd.DataFrame         # rebalance date → ticker weights
    turnover_history: pd.Series           # rebalance date → one-way turnover
    risk_history: pd.DataFrame            # rebalance date → risk metrics
    factor_exposure_history: pd.DataFrame # rebalance date → factor exposures

    # derived performance metrics (computed on init)
    cumulative_return: float = field(init=False)
    annualised_return: float = field(init=False)
    annualised_vol: float = field(init=False)
    sharpe_ratio: float = field(init=False)
    max_drawdown: float = field(init=False)
    calmar_ratio: float = field(init=False)

    def __post_init__(self):
        r = self.portfolio_returns.dropna()
        T = len(r)
        self.cumulative_return = float(np.exp(r.sum()) - 1)
        self.annualised_return = float(np.exp(r.mean() * 252) - 1)
        self.annualised_vol = float(r.std() * np.sqrt(252))
        self.sharpe_ratio = self.annualised_return / (self.annualised_vol + 1e-9)

        cum_log = r.cumsum()
        rolling_max = cum_log.cummax()
        drawdown = (cum_log - rolling_max)
        self.max_drawdown = float(drawdown.min())
        self.calmar_ratio = self.annualised_return / (abs(self.max_drawdown) + 1e-9)

    def summary(self) -> Dict:
        return {
            "cumulative_return": round(self.cumulative_return, 4),
            "annualised_return": round(self.annualised_return, 4),
            "annualised_vol": round(self.annualised_vol, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "mean_turnover": round(float(self.turnover_history.mean()), 4),
        }


class BacktestEngine:
    """
    Runs a walk-forward factor backtest.

    Parameters
    ----------
    prices       : adjusted close prices (T × N)
    returns      : log returns (T × N), aligned to prices
    factors      : list of BaseFactor instances to combine
    factor_weights : name → weight for composite score (equal if None)
    lookback     : rolling window (days) for factor computation and risk model fit
    rebalance_freq : 'monthly' | 'weekly' | int (every N trading days)
    optimizer_mode : 'max_sharpe' | 'min_variance' | 'risk_parity' | 'factor_tilt'
    risk_budget   : annualised vol budget for factor_tilt mode
    max_weight    : per-asset weight cap
    long_only     : True for long-only, False for long-short
    turnover_penalty : L1 turnover cost in objective
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        factors: List[BaseFactor],
        factor_weights: Optional[Dict[str, float]] = None,
        lookback: int = 252,
        rebalance_freq: str = "monthly",
        optimizer_mode: str = "max_sharpe",
        risk_budget: float = 0.15,
        max_weight: float = 0.10,
        long_only: bool = True,
        turnover_penalty: float = 0.001,
        universe_selector=None,
    ):
        self.prices = prices
        self.returns = returns
        self.factors = factors
        self.factor_weights = factor_weights
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.optimizer_mode = optimizer_mode
        self.risk_budget = risk_budget
        self.max_weight = max_weight
        self.long_only = long_only
        self.turnover_penalty = turnover_penalty
        # Optional callable: (rebal_date, price_window, ret_window) -> [ticker, ...]
        self.universe_selector = universe_selector

    def run(self) -> BacktestResult:
        rebalance_dates = self._rebalance_dates()
        all_dates = self.returns.index

        port_ret = pd.Series(0.0, index=all_dates, dtype=float)
        weights_hist: Dict[pd.Timestamp, pd.Series] = {}
        turnover_hist: Dict[pd.Timestamp, float] = {}
        risk_hist: List[Dict] = []
        exposure_hist: List[Dict] = []

        current_weights: Optional[pd.Series] = None

        for i, rebal_date in enumerate(rebalance_dates):
            window_start_idx = all_dates.get_loc(rebal_date) - self.lookback
            if window_start_idx < 0:
                logger.debug("Skipping %s: insufficient history", rebal_date)
                continue

            window_dates = all_dates[window_start_idx: all_dates.get_loc(rebal_date) + 1]
            ret_window = self.returns.loc[window_dates]
            price_window = self.prices.loc[window_dates]

            # drop tickers with any NaN in the window
            valid_cols = ret_window.columns[ret_window.notna().all()]

            # apply dynamic universe selection (e.g. top-N by dollar volume)
            if self.universe_selector is not None:
                selected = self.universe_selector(rebal_date, price_window[valid_cols], ret_window[valid_cols])
                valid_cols = valid_cols.intersection(selected)
            ret_window = ret_window[valid_cols]
            price_window = price_window[valid_cols]

            if len(valid_cols) < 5:
                logger.warning("Too few valid tickers at %s, skipping", rebal_date)
                continue

            # 1. Compute and combine factors
            factor_scores: Dict[str, pd.DataFrame] = {}
            for f in self.factors:
                raw = f.compute(price_window, ret_window)
                factor_scores[f.name] = cross_sectional_zscore(raw.fillna(0))

            composite = composite_score(factor_scores, self.factor_weights)
            latest_scores = composite.iloc[-1].dropna()

            if len(latest_scores) < 5:
                continue

            # 2. Fit risk model
            risk_model = FactorRiskModel(ret_window, factor_scores, use_scores=True)
            risk_model.fit()
            cov = risk_model.total_covariance()

            # align scores and cov
            common = latest_scores.index.intersection(cov.index)
            scores_aligned = latest_scores.loc[common]
            cov_aligned = cov.loc[common, common]

            # 3. Optimise
            opt = PortfolioOptimizer(scores_aligned, cov_aligned)
            result: OptimizationResult = self._run_optimizer(
                opt, current_weights
            )
            new_weights = result.weights

            # 4. Record turnover
            if current_weights is not None:
                prev = current_weights.reindex(new_weights.index).fillna(0)
                turnover = float((new_weights - prev).abs().sum() / 2)
            else:
                turnover = 1.0
            turnover_hist[rebal_date] = turnover

            # 5. Risk attribution at rebalance
            attr = risk_model.attribute(new_weights)
            risk_hist.append({"date": rebal_date, **attr.to_dict()})
            exposure_hist.append({
                "date": rebal_date,
                **{f"exposure_{k}": v for k, v in attr.factor_contributions.items()},
            })

            weights_hist[rebal_date] = new_weights
            current_weights = new_weights

            # 6. Apply weights until next rebalance
            next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else all_dates[-1]
            hold_dates = all_dates[
                (all_dates > rebal_date) & (all_dates <= next_date)
            ]
            w = new_weights.reindex(self.returns.columns).fillna(0)
            port_ret.loc[hold_dates] = (self.returns.loc[hold_dates] @ w).values

        weights_df = pd.DataFrame(weights_hist).T.fillna(0)
        turnover_s = pd.Series(turnover_hist, name="turnover")
        risk_df = pd.DataFrame(risk_hist).set_index("date") if risk_hist else pd.DataFrame()
        exposure_df = pd.DataFrame(exposure_hist).set_index("date") if exposure_hist else pd.DataFrame()

        return BacktestResult(
            portfolio_returns=port_ret,
            weights_history=weights_df,
            turnover_history=turnover_s,
            risk_history=risk_df,
            factor_exposure_history=exposure_df,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebalance_dates(self) -> List[pd.Timestamp]:
        dates = self.returns.index
        if self.rebalance_freq == "monthly":
            monthly = dates.to_period("M").drop_duplicates()
            return [dates[dates.to_period("M") == m][-1] for m in monthly]
        elif self.rebalance_freq == "weekly":
            weekly = dates.to_period("W").drop_duplicates()
            return [dates[dates.to_period("W") == w][-1] for w in weekly]
        elif isinstance(self.rebalance_freq, int):
            step = self.rebalance_freq
            return list(dates[::step])
        raise ValueError(f"Unknown rebalance_freq: {self.rebalance_freq}")

    def _run_optimizer(
        self, opt: PortfolioOptimizer, prev_weights: Optional[pd.Series]
    ) -> OptimizationResult:
        try:
            if self.optimizer_mode == "max_sharpe":
                return opt.max_sharpe(
                    long_only=self.long_only,
                    max_weight=self.max_weight,
                    turnover_penalty=self.turnover_penalty,
                    prev_weights=prev_weights,
                )
            elif self.optimizer_mode == "min_variance":
                return opt.min_variance(
                    long_only=self.long_only,
                    max_weight=self.max_weight,
                )
            elif self.optimizer_mode == "risk_parity":
                return opt.risk_parity()
            elif self.optimizer_mode == "factor_tilt":
                return opt.factor_tilt(
                    risk_budget=self.risk_budget,
                    long_only=self.long_only,
                    max_weight=self.max_weight,
                )
        except Exception as e:
            logger.warning("Optimiser failed (%s), falling back to equal weight", e)
        return opt._fallback_equal_weight("optimiser exception")
