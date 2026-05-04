"""
Factor research utilities: IC, ICIR, decay, quintile analysis, turnover.
All methods operate on (dates × tickers) DataFrames.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FactorStats:
    name: str
    mean_ic: float
    icir: float
    ic_series: pd.Series
    decay_profile: pd.Series          # horizon → mean IC
    quintile_returns: pd.DataFrame    # horizon × quintile
    mean_turnover: float
    summary: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.summary = {
            "mean_ic": round(self.mean_ic, 4),
            "icir": round(self.icir, 4),
            "mean_turnover": round(self.mean_turnover, 4),
            "t_stat": round(self.mean_ic / (self.ic_series.std() / np.sqrt(len(self.ic_series)) + 1e-9), 4),
        }


class FactorResearch:
    """
    Evaluates predictive quality of a single factor DataFrame.

    Parameters
    ----------
    factor : pd.DataFrame
        Cross-sectionally normalised factor scores (dates × tickers).
    returns : pd.DataFrame
        Forward daily log returns, aligned to factor dates and tickers.
    """

    def __init__(self, factor: pd.DataFrame, returns: pd.DataFrame):
        common_idx = factor.index.intersection(returns.index)
        common_cols = factor.columns.intersection(returns.columns)
        self.factor = factor.loc[common_idx, common_cols]
        self.returns = returns.loc[common_idx, common_cols]

    # ------------------------------------------------------------------
    # Information Coefficient
    # ------------------------------------------------------------------

    def ic(self, forward_period: int = 1, method: str = "spearman") -> pd.Series:
        """
        Compute daily cross-sectional IC between factor scores and
        `forward_period`-day ahead returns.

        method: 'spearman' (rank IC) or 'pearson'
        """
        if method not in ("spearman", "pearson"):
            raise ValueError(f"method must be 'spearman' or 'pearson', got '{method}'")
        fwd = self.returns.shift(-forward_period)
        rank_fn = stats.spearmanr if method == "spearman" else stats.pearsonr

        ic_values = []
        dates = []
        for date, row in self.factor.iterrows():
            fwd_row = fwd.loc[date]
            valid = row.notna() & fwd_row.notna()
            if valid.sum() < 5:
                continue
            f_vals, r_vals = row[valid].values, fwd_row[valid].values
            # skip constant inputs — correlation is undefined
            if f_vals.std() < 1e-12 or r_vals.std() < 1e-12:
                continue
            corr, _ = rank_fn(f_vals, r_vals)
            if np.isnan(corr):
                continue
            ic_values.append(corr)
            dates.append(date)

        return pd.Series(ic_values, index=pd.DatetimeIndex(dates), name="IC")

    def icir(self, forward_period: int = 1) -> float:
        """IC Information Ratio = mean(IC) / std(IC)."""
        ic_series = self.ic(forward_period)
        return float(ic_series.mean() / (ic_series.std() + 1e-9))

    # ------------------------------------------------------------------
    # Factor Decay
    # ------------------------------------------------------------------

    def decay_profile(self, max_horizon: int = 20) -> pd.Series:
        """
        Mean IC across horizons 1..max_horizon, computed in parallel.
        A fast-decaying factor is better suited for short holding periods.
        """
        horizons = list(range(1, max_horizon + 1))
        # Each horizon's IC is independent — safe to compute concurrently.
        with ThreadPoolExecutor(max_workers=min(max_horizon, 8)) as ex:
            mean_ics = dict(zip(horizons, ex.map(lambda h: float(self.ic(h).mean()), horizons)))
        return pd.Series(mean_ics, name="mean_IC")

    # ------------------------------------------------------------------
    # Quintile Analysis
    # ------------------------------------------------------------------

    def quintile_returns(
        self, forward_periods: List[int] = None, n_quintiles: int = 5
    ) -> pd.DataFrame:
        """
        For each forward_period, bucket stocks into n_quintiles by factor score
        and compute mean annualised return per quintile.

        Returns DataFrame indexed by forward_period, columns Q1..Q5.
        """
        if forward_periods is None:
            forward_periods = [1, 5, 10, 21]

        labels = [f"Q{i+1}" for i in range(n_quintiles)]
        result = {}

        for horizon in forward_periods:
            fwd = self.returns.rolling(horizon).sum().shift(-horizon)
            ann = fwd * (252 / horizon)

            quintile_means = {q: [] for q in labels}

            for date, scores in self.factor.iterrows():
                fwd_row = ann.loc[date]
                valid = scores.notna() & fwd_row.notna()
                if valid.sum() < n_quintiles:
                    continue
                s_valid = scores[valid]
                if s_valid.std() < 1e-12:
                    continue
                buckets = pd.qcut(s_valid, n_quintiles, labels=labels, duplicates="drop")
                if buckets.nunique() < n_quintiles:
                    continue
                for q in labels:
                    members = buckets[buckets == q].index
                    quintile_means[q].append(fwd_row[members].mean())

            result[horizon] = {q: np.nanmean(quintile_means[q]) for q in labels}

        return pd.DataFrame(result, columns=list(result.keys())).T.rename_axis("horizon")

    # ------------------------------------------------------------------
    # Turnover
    # ------------------------------------------------------------------

    def turnover(self, top_pct: float = 0.2) -> pd.Series:
        """
        Daily turnover of the long portfolio (top `top_pct` fraction by score).
        Turnover = fraction of the long book that changes each day.
        """
        n = max(1, int(self.factor.shape[1] * top_pct))
        longs_prev = None
        turnovers = []
        dates = []

        for date, scores in self.factor.iterrows():
            valid = scores.dropna()
            if len(valid) < n:
                longs_prev = None
                continue
            longs = set(valid.nlargest(n).index)
            if longs_prev is not None:
                changed = len(longs.symmetric_difference(longs_prev))
                turnovers.append(changed / (2 * n))
                dates.append(date)
            longs_prev = longs

        return pd.Series(turnovers, index=pd.DatetimeIndex(dates), name="turnover")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def evaluate(self, name: str = "factor", forward_period: int = 1) -> FactorStats:
        """Run all diagnostics and return a FactorStats summary."""
        ic_series = self.ic(forward_period)
        return FactorStats(
            name=name,
            mean_ic=float(ic_series.mean()),
            icir=self.icir(forward_period),
            ic_series=ic_series,
            decay_profile=self.decay_profile(max_horizon=20),
            quintile_returns=self.quintile_returns(forward_periods=[1, 5, 21]),
            mean_turnover=float(self.turnover().mean()),
        )
