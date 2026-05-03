import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from factor_alpha import MomentumFactor, LowVolatilityFactor, cross_sectional_zscore
from factor_alpha.analysis.factor_research import FactorResearch

# ── Synthetic data (replace with real prices/returns) ───────────────────────
rng   = np.random.default_rng(42)
dates = pd.date_range("2020-01-02", periods=500, freq="B")
tickers = [f"T{i:02d}" for i in range(30)]
returns = pd.DataFrame(rng.normal(0, 0.01, (500, 30)), index=dates, columns=tickers)
prices  = 100 * np.exp(returns.cumsum())

# ── Build a factor score ───────────────────────────────────────────────────��─
mom_raw   = MomentumFactor(lookback=60, skip=5).compute(prices, returns)
mom_score = cross_sectional_zscore(mom_raw)

# ── 1. Construct ─────────────────────────────────────────────────────────────
research = FactorResearch(factor=mom_score, returns=returns)

# ── 2. IC — daily cross-sectional rank correlation with 1-day forward return ─
ic_daily = research.ic(forward_period=1, method="spearman")
print(f"Mean IC (1d):  {ic_daily.mean():.4f}")
print(f"IC Std:        {ic_daily.std():.4f}")

# Monthly IC (21 trading days forward)
ic_monthly = research.ic(forward_period=21)
print(f"Mean IC (21d): {ic_monthly.mean():.4f}")

# ── 3. ICIR — information ratio of the IC series ─────────────────────────────
icir = research.icir(forward_period=1)
print(f"ICIR: {icir:.4f}")   # > 0.5 is considered strong

# ── 4. Decay profile — mean IC across horizons 1–20 days ─────────────────────
decay = research.decay_profile(max_horizon=20)
print(decay)
# horizon
# 1     0.0123
# 2     0.0098
# ...

# ── 5. Quintile returns — annualised return per score quintile ────────────────
quintiles = research.quintile_returns(
    forward_periods=[1, 5, 21],   # 1-day, 1-week, 1-month
    n_quintiles=5,
)
print(quintiles)
# horizon   Q1      Q2      Q3      Q4      Q5       ← Q5 = highest scores
#       1  -0.03   0.01   0.02   0.03   0.05        spread = Q5 - Q1

# ── 6. Turnover — how much the long book changes each day ─────────────────────
turnover = research.turnover(top_pct=0.2)   # top 20% = 6 stocks
print(f"Mean daily turnover: {turnover.mean():.2%}")

# ── 7. Full summary via evaluate() ───────────────────────────────────────────
stats = research.evaluate(name="momentum_60_5", forward_period=21)
print(stats.summary)
# {'mean_ic': 0.0231, 'icir': 0.41, 'mean_turnover': 0.18, 't_stat': 2.3}

# ── 8. Plot IC time series + decay profile ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ic_daily.rolling(21).mean().plot(ax=axes[0], color="steelblue")
axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
axes[0].set_title("IC (21-day rolling mean)")
axes[0].set_ylabel("Spearman IC")

decay.plot(kind="bar", ax=axes[1], color="steelblue", alpha=0.7)
axes[1].set_title("Factor Decay Profile")
axes[1].set_xlabel("Forward horizon (days)")
axes[1].set_ylabel("Mean IC")

spread = quintiles["Q5"] - quintiles["Q1"]
spread.plot(kind="bar", ax=axes[2], color="seagreen", alpha=0.7)
axes[2].set_title("Q5 - Q1 Return Spread (Ann.)")
axes[2].set_xlabel("Forward horizon (days)")
axes[2].set_ylabel("Ann. return spread")

plt.tight_layout()
plt.savefig("factor_research.png", dpi=150, bbox_inches="tight")
plt.show()
