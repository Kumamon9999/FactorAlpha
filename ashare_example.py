"""
A-share 5-minute ZIP data → FactorAlpha full pipeline example.

Dataset layout:
  dataset/
    YYYY-MM/
      YYYYMMDD_5min.zip   (one ZIP per trading day, ~5100 stocks inside)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from factor_alpha.data import AShareZipLoader
from factor_alpha import (
    MomentumFactor, LowVolatilityFactor,
    cross_sectional_zscore, composite_score,
    BacktestEngine,
    FactorResearch,
)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "src/factor_alpha/data/dataset")

# ── 1. Pick a liquid A-share universe ────────────────────────────────────────
# Using a representative selection across sectors. Tickers follow the format:
#   sh6XXXXX  → Shanghai Main Board
#   sz0XXXXX  → Shenzhen Main Board / SME
#   sz3XXXXX  → ChiNext (创业板)
ASHARE_POOL = [
    # financials
    "sh601318", "sh601288", "sh600036", "sh601166", "sh600016",  # Ping An, ABC, CMB, IB, Minsheng
    # consumer / liquor
    "sh600519", "sz000858", "sh600809", "sz000568",              # Moutai, Wuliangye, Fenjiu, Luzhou
    # energy / materials
    "sh600028", "sh601857", "sh600900", "sh601088",              # Sinopec, PetroChina, Yangtze Power, Shenhua
    # tech / semiconductor
    "sh688981", "sh688012", "sz002415", "sz300059",              # SMIC, SiEn, Hikvision, East Money
    # pharma / biotech
    "sh600276", "sz000661", "sh688180", "sz300760",              # Hengrui, Cansino→replace, Junshi, Mindray
    # industrials / EV
    "sz002594", "sh601899", "sz300750", "sh600104",              # BYD, Zijin, CATL, SAIC
    # consumer staples / retail
    "sz000651", "sh600887", "sz002352",                          # Gree, Yili, S.F. Express
]

# ── 2. Load and resample 5-min → daily ───────────────────────────────────────
print("Loading A-share 5-min data from ZIPs ...")
loader = AShareZipLoader(
    dataset_dir=DATASET_DIR,
    tickers=ASHARE_POOL,
    start="2025-01-01",
    end="2025-12-31",
    missing_threshold=0.2,
)

prices  = loader.prices          # daily close prices  (dates × tickers)
volumes = loader.volumes         # daily share volume
amounts = loader.amounts         # daily turnover (RMB) — better liquidity proxy
returns = loader.get_returns(method="log")

print(f"Loaded: {prices.shape[1]} tickers × {len(prices)} trading days")
print(f"Date range: {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"Tickers: {prices.columns.tolist()}\n")

# ── 3. Rolling universe selector (top-N by turnover amount) ──────────────────
# At each rebalance, pick the top 20 most-traded stocks from the pool.
# Uses only information available on that date → no look-ahead bias.
dollar_vol = loader.get_dollar_volume().reindex(returns.index)
UNIVERSE_SIZE = 20

def ashare_selector(rebal_date, price_window, ret_window):
    dv = dollar_vol.loc[:rebal_date].iloc[-21:]
    eligible = ret_window.columns.intersection(dv.columns)
    return dv[eligible].mean().nlargest(UNIVERSE_SIZE).index

# ── 4. Compute factor scores ──────────────────────────────────────────────────
print("Computing factor scores ...")
mom    = MomentumFactor(lookback=60, skip=5)
lowvol = LowVolatilityFactor(lookback=40)

scores_mom    = cross_sectional_zscore(mom.compute(prices, returns))
scores_lowvol = cross_sectional_zscore(lowvol.compute(prices, returns))

combined = composite_score(
    {"momentum": scores_mom, "lowvol": scores_lowvol},
    weights={"momentum": 0.6, "lowvol": 0.4},
)

# ── 5. Factor quality diagnostics ────────────────────────────────────────────
print("Running factor research ...")
research = FactorResearch(factor=combined, returns=returns)
stats    = research.evaluate(name="mom_lowvol", forward_period=5)
print("\nFactor Summary (5-day forward IC):")
for k, v in stats.summary.items():
    print(f"  {k:<18}: {v}")

# ── 6. Backtest ───────────────────────────────────────────────────────────────
print("\nRunning backtest ...")
engine = BacktestEngine(
    prices=prices,
    returns=returns,
    factors=[MomentumFactor(lookback=60, skip=5),
             LowVolatilityFactor(lookback=40)],
    lookback=60,
    rebalance_freq="monthly",
    optimizer_mode="max_sharpe",
    max_weight=0.15,
    turnover_penalty=0.002,
    long_only=True,
    universe_selector=ashare_selector,
)
result = engine.run()
print("\nBacktest Summary:")
for k, v in result.summary().items():
    print(f"  {k:<22}: {v}")

# ── 7. Plot dashboard ─────────────────────────────────────────────────────────
r       = result.portfolio_returns.dropna()
cum_ret = np.exp(r.cumsum()) - 1
dd_log  = r.cumsum()
drawdown = dd_log - dd_log.cummax()

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# cumulative return
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum_ret * 100, color="firebrick", linewidth=1.5, label="Strategy")
ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax1.set_title("A-Share Strategy — Cumulative Return (%)")
ax1.set_ylabel("%")
ax1.legend()
ax1.grid(True, alpha=0.3)

# drawdown
ax2 = fig.add_subplot(gs[1, :])
ax2.fill_between(drawdown.index, drawdown * 100, 0, color="firebrick", alpha=0.4)
ax2.set_title("Drawdown (%)")
ax2.set_ylabel("%")
ax2.grid(True, alpha=0.3)

# rolling vol
ax3 = fig.add_subplot(gs[2, 0])
(r.rolling(20).std() * np.sqrt(252) * 100).plot(ax=ax3, color="darkorange", linewidth=1.2)
ax3.set_title("Rolling 20-day Ann. Vol (%)")
ax3.set_ylabel("%")
ax3.grid(True, alpha=0.3)

# turnover
ax4 = fig.add_subplot(gs[2, 1])
if len(result.turnover_history) > 0:
    ax4.bar(result.turnover_history.index,
            result.turnover_history * 100, width=12, color="slategray", alpha=0.7)
ax4.set_title("One-Way Turnover per Rebalance (%)")
ax4.set_ylabel("%")
ax4.grid(True, alpha=0.3)

summary = result.summary()
ann = (
    f"Ann. Return: {summary['annualised_return']*100:.1f}%\n"
    f"Ann. Vol:    {summary['annualised_vol']*100:.1f}%\n"
    f"Sharpe:      {summary['sharpe_ratio']:.2f}\n"
    f"Max DD:      {summary['max_drawdown']*100:.1f}%\n"
    f"Calmar:      {summary['calmar_ratio']:.2f}"
)
fig.text(0.76, 0.96, ann, transform=fig.transFigure, fontsize=9,
         verticalalignment="top", family="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.suptitle("FactorAlpha — A-Share Momentum + LowVol (Max Sharpe)", fontsize=13)
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
out = os.path.join(PLOT_DIR, "ashare_backtest.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {out}")
