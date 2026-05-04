import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from factor_alpha import MomentumFactor, LowVolatilityFactor, cross_sectional_zscore
from factor_alpha.factors.ml import (
    PCAResidualFactor,
    RidgeAlphaFactor,
    GradientBoostFactor,
    RandomForestFactor,
)
from factor_alpha.analysis.factor_research import FactorResearch

# ── Synthetic data ────────────────────────────────────────────────────────────
# 600 days × 40 stocks. We inject a weak momentum signal so ML factors have
# something to learn: each stock's return has a small autocorrelation component.
rng     = np.random.default_rng(42)
dates   = pd.date_range("2020-01-02", periods=600, freq="B")
tickers = [f"T{i:02d}" for i in range(40)]

noise   = rng.normal(0, 0.01, (600, 40))
# AR(1) with rho=0.05 so lagged returns carry a tiny signal
ar_ret  = np.zeros_like(noise)
ar_ret[0] = noise[0]
for t in range(1, 600):
    ar_ret[t] = 0.05 * ar_ret[t - 1] + noise[t]

returns = pd.DataFrame(ar_ret, index=dates, columns=tickers)
prices  = 100 * np.exp(returns.cumsum())

print("=" * 60)
print("Computing factor scores ...")
print("=" * 60)

# ── Traditional baseline ──────────────────────────────────────────────────────
mom_score   = cross_sectional_zscore(MomentumFactor(lookback=60, skip=5).compute(prices, returns))
lowvol_score = cross_sectional_zscore(LowVolatilityFactor(lookback=40).compute(prices, returns))

# ── ML factors ───────────────────────────────────────────────────────────────
pca_score  = cross_sectional_zscore(PCAResidualFactor(n_components=5, signal_window=21).compute(prices, returns))
ridge_score = cross_sectional_zscore(RidgeAlphaFactor(forward=1, alpha=1.0, min_obs=63).compute(prices, returns))
gbm_score   = cross_sectional_zscore(GradientBoostFactor(forward=1, n_estimators=50, max_depth=3, min_obs=63).compute(prices, returns))
rf          = RandomForestFactor(forward=1, n_estimators=50, max_depth=4, min_obs=63)
rf_score    = cross_sectional_zscore(rf.compute(prices, returns))

factors = {
    "Momentum (trad)":    mom_score,
    "LowVol (trad)":      lowvol_score,
    "PCA Residual (ml)":  pca_score,
    "Ridge Alpha (ml)":   ridge_score,
    "GradBoost (ml)":     gbm_score,
    "RandomForest (ml)":  rf_score,
}

# ── Evaluate each factor with FactorResearch ──────────────────────────────────
print(f"\n{'Factor':<22} {'Mean IC':>9} {'ICIR':>8} {'t-stat':>8} {'Turnover':>10}")
print("-" * 62)

results = {}
for name, score in factors.items():
    r = FactorResearch(factor=score.dropna(how="all"), returns=returns)
    stats = r.evaluate(name=name, forward_period=5)
    results[name] = {"research": r, "stats": stats}
    s = stats.summary
    print(f"{name:<22} {s['mean_ic']:>9.4f} {s['icir']:>8.4f} {s['t_stat']:>8.4f} {s['mean_turnover']:>10.4f}")

# ── Random Forest feature importances ────────────────────────────────────────
if hasattr(rf, "feature_importances_"):
    print("\nRandomForest feature importances:")
    for feat, imp in sorted(rf.feature_importances_.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 200)
        print(f"  {feat:<12} {imp:.4f}  {bar}")

# ── Plot dashboard ────────────────────────────────────────────────────────────
COLORS = {
    "Momentum (trad)":    ("steelblue",   "-"),
    "LowVol (trad)":      ("slategray",   "-"),
    "PCA Residual (ml)":  ("darkorange",  "--"),
    "Ridge Alpha (ml)":   ("seagreen",    "--"),
    "GradBoost (ml)":     ("crimson",     "--"),
    "RandomForest (ml)":  ("mediumpurple","--"),
}

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: IC rolling mean (one panel per factor)
ax_ic = fig.add_subplot(gs[0, :])
for name, v in results.items():
    ic = v["research"].ic(forward_period=5)
    if ic.empty or ic.dropna().empty:
        continue
    color, ls = COLORS[name]
    ic.rolling(21).mean().plot(ax=ax_ic, label=name, color=color, linestyle=ls, linewidth=1.2)
ax_ic.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax_ic.set_title("IC — 21-day Rolling Mean (5-day forward)")
ax_ic.set_ylabel("Spearman IC")
ax_ic.legend(fontsize=8, ncol=3)
ax_ic.grid(True, alpha=0.3)

# Row 1 left: Mean IC bar chart
ax_bar = fig.add_subplot(gs[1, 0])
mean_ics = {n: v["stats"].summary["mean_ic"] for n, v in results.items()}
bars = ax_bar.bar(range(len(mean_ics)), list(mean_ics.values()),
                  color=[COLORS[n][0] for n in mean_ics])
ax_bar.set_xticks(range(len(mean_ics)))
ax_bar.set_xticklabels([n.split(" ")[0] for n in mean_ics], rotation=30, ha="right", fontsize=8)
ax_bar.axhline(0, color="black", linewidth=0.5)
ax_bar.set_title("Mean IC (5d forward)")
ax_bar.grid(True, alpha=0.3, axis="y")

# Row 1 middle: ICIR bar chart
ax_icir = fig.add_subplot(gs[1, 1])
icirs = {n: v["stats"].summary["icir"] for n, v in results.items()}
ax_icir.bar(range(len(icirs)), list(icirs.values()),
            color=[COLORS[n][0] for n in icirs])
ax_icir.set_xticks(range(len(icirs)))
ax_icir.set_xticklabels([n.split(" ")[0] for n in icirs], rotation=30, ha="right", fontsize=8)
ax_icir.axhline(0, color="black", linewidth=0.5)
ax_icir.set_title("ICIR")
ax_icir.grid(True, alpha=0.3, axis="y")

# Row 1 right: Mean turnover
ax_to = fig.add_subplot(gs[1, 2])
turnovers = {n: v["stats"].summary["mean_turnover"] for n, v in results.items()}
ax_to.bar(range(len(turnovers)), list(turnovers.values()),
          color=[COLORS[n][0] for n in turnovers])
ax_to.set_xticks(range(len(turnovers)))
ax_to.set_xticklabels([n.split(" ")[0] for n in turnovers], rotation=30, ha="right", fontsize=8)
ax_to.set_title("Mean Daily Turnover")
ax_to.grid(True, alpha=0.3, axis="y")

# Row 2: Decay profiles
ax_decay = fig.add_subplot(gs[2, :2])
for name, v in results.items():
    color, ls = COLORS[name]
    v["stats"].decay_profile.plot(ax=ax_decay, label=name, color=color, linestyle=ls, linewidth=1.2)
ax_decay.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax_decay.set_title("Factor Decay Profile (Mean IC by Horizon)")
ax_decay.set_xlabel("Forward horizon (days)")
ax_decay.set_ylabel("Mean IC")
ax_decay.legend(fontsize=8)
ax_decay.grid(True, alpha=0.3)

# Row 2 right: Q5-Q1 spread at 5-day horizon for all factors
ax_spread = fig.add_subplot(gs[2, 2])
spreads = {}
for name, v in results.items():
    q = v["stats"].quintile_returns
    if 5 in q.index:
        spreads[name] = q.loc[5, "Q5"] - q.loc[5, "Q1"]
if spreads:
    ax_spread.bar(range(len(spreads)), list(spreads.values()),
                  color=[COLORS[n][0] for n in spreads])
    ax_spread.set_xticks(range(len(spreads)))
    ax_spread.set_xticklabels([n.split(" ")[0] for n in spreads], rotation=30, ha="right", fontsize=8)
    ax_spread.axhline(0, color="black", linewidth=0.5)
    ax_spread.set_title("Q5-Q1 Return Spread (5d, Ann.)")
    ax_spread.grid(True, alpha=0.3, axis="y")

plt.suptitle("Factor Research: Traditional vs ML Factors", fontsize=14)
PLOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "plots"))
os.makedirs(PLOT_DIR, exist_ok=True)
out = os.path.join(PLOT_DIR, "factor_research_ml.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {out}")
