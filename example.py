import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from factor_alpha import (
    MomentumFactor, LowVolatilityFactor,
    cross_sectional_zscore, composite_score,
    BacktestEngine,
    FactorResearch,
)

# ── 1. Load data ────────────────────────────────────────────────────────────
# Broad Nasdaq candidate pool (~150 stocks). At each monthly rebalance the
# engine calls ndx100_selector() which picks the top-100 by trailing 21-day
# average dollar volume — a point-in-time proxy for NDX100 membership that
# uses only information available on that date (no look-ahead bias).
NASDAQ_POOL = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "QCOM", "TXN", "INTC", "AMD", "MU", "AMAT", "LRCX", "ADI", "KLAC", "MRVL", "ON",
    "ADBE", "INTU", "CSCO", "PANW", "FTNT", "CDNS", "SNPS",
    "WDAY", "CRWD", "DDOG", "ZS", "TEAM", "TTD", "OKTA", "SNOW", "NET",
    "NFLX", "COST", "MELI", "BIDU", "JD", "PDD", "ASML",
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "IDXX", "DXCM", "ILMN", "ISRG",
    "PEP", "MDLZ", "MNST", "KDP",
    "ADP", "PAYX", "FAST", "CTAS", "ODFL", "PCAR", "EXC", "XEL",
    "ROST", "ORLY", "CPRT", "FANG", "CEG", "GEHC",
    "MCHP", "NXPI", "SWKS", "QRVO", "MPWR", "ENTG",
    "ZM", "DOCU", "SPLK", "VEEV", "ANSS", "MRNA", "BKNG", "EXPE",
]
# deduplicate in case of accidental repeats
NASDAQ_POOL = list(dict.fromkeys(NASDAQ_POOL))

START, END = "2018-01-01", "2026-05-01"
BATCH, PAUSE = 10, 2.0

def _fetch_batched(tickers, start, end):
    price_chunks, volume_chunks = [], []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i: i + BATCH]
        print(f"Fetching batch {i//BATCH + 1}: {batch}")
        raw = yf.download(batch, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            price_chunks.append(raw["Close"])
            volume_chunks.append(raw["Volume"])
        else:
            price_chunks.append(raw[["Close"]].rename(columns={"Close": batch[0]}))
            volume_chunks.append(raw[["Volume"]].rename(columns={"Volume": batch[0]}))
        time.sleep(PAUSE)
    return pd.concat(price_chunks, axis=1), pd.concat(volume_chunks, axis=1)

raw_prices, raw_volumes = _fetch_batched(NASDAQ_POOL, START, END)
raw_prices  = raw_prices.ffill().bfill()
raw_volumes = raw_volumes.ffill().bfill()

valid = raw_prices.notna().mean() >= 0.8
dropped = valid[~valid].index.tolist()
if dropped:
    print(f"Dropping tickers with insufficient data: {dropped}")
prices   = raw_prices.loc[:, valid]
volumes  = raw_volumes.loc[:, valid]
returns  = np.log(prices / prices.shift(1)).iloc[1:]
volumes  = volumes.iloc[1:]   # align index with returns

# dollar volume DataFrame used by the universe selector
dollar_vol = (prices * volumes).iloc[1:]

# ── 1b. Rolling universe selector ──────────────────────────────────────────
NDX_SIZE = 100   # simulate NDX100 size

def ndx100_selector(rebal_date, price_window, ret_window):
    """Return the top NDX_SIZE tickers by trailing 21-day avg dollar volume."""
    dv_window = dollar_vol.loc[:rebal_date].iloc[-21:]
    dv_window = dv_window.loc[:, ~dv_window.columns.duplicated()]
    eligible = ret_window.columns.intersection(dv_window.columns)
    avg_dv = dv_window[eligible].mean()
    return avg_dv.nlargest(NDX_SIZE).index

# ── 2. Compute factor scores ────────────────────────────────────────────────
mom   = MomentumFactor(lookback=252, skip=21)
lowvol = LowVolatilityFactor(lookback=60)

scores_mom    = cross_sectional_zscore(mom.compute(prices, returns))
scores_lowvol = cross_sectional_zscore(lowvol.compute(prices, returns))

# Combine with equal weight
combined = composite_score(
    {"momentum": scores_mom, "lowvol": scores_lowvol},
    weights={"momentum": 0.5, "lowvol": 0.5},
)

# ── 3. Analyse factor quality ───────────────────────────────────────────────
research = FactorResearch(factor=combined, returns=returns)
stats = research.evaluate(name="combined", forward_period=21)
print(stats.summary)

# ── 4. Run a backtest ───────────────────────────────────────────────────────
engine = BacktestEngine(
    prices=prices,
    returns=returns,
    factors=[MomentumFactor(lookback=252, skip=21),
             LowVolatilityFactor(lookback=60)],
    lookback=252,
    rebalance_freq="monthly",
    optimizer_mode="max_sharpe",
    max_weight=0.10,
    turnover_penalty=0.001,
    long_only=True,
    universe_selector=ndx100_selector,
)
result = engine.run()

# ── 5. Inspect results ──────────────────────────────────────────────────────
print(result.summary())

# ── 6. Fetch Nasdaq-100 (QQQ) baseline ─────────────────────────────────────
r = result.portfolio_returns.dropna()

qqq_raw = yf.download("QQQ", start=START, end=END,
                      auto_adjust=True, progress=False)
qqq_prices = qqq_raw["Close"].squeeze()
qqq_log_ret = np.log(qqq_prices / qqq_prices.shift(1)).dropna()

# align to portfolio date range
common_idx = r.index.intersection(qqq_log_ret.index)
qqq_cum = np.exp(qqq_log_ret.loc[common_idx].cumsum()) - 1
qqq_dd_log = qqq_log_ret.loc[common_idx].cumsum()
qqq_dd = qqq_dd_log - qqq_dd_log.cummax()
qqq_roll_vol = qqq_log_ret.loc[common_idx].rolling(63).std() * np.sqrt(252) * 100

# ── 7. Plot backtest dashboard ──────────────────────────────────────────────
cum_log = r.cumsum()
cum_ret = np.exp(cum_log) - 1
rolling_max = cum_log.cummax()
drawdown = cum_log - rolling_max

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# 1. Cumulative return
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(cum_ret.reindex(common_idx) * 100, color="steelblue", linewidth=1.5, label="Strategy")
ax1.plot(qqq_cum * 100, color="darkorange", linewidth=1.5, linestyle="--", label="Nasdaq-100 (QQQ)")
ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax1.set_title("Cumulative Return (%)")
ax1.set_ylabel("%")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Drawdown
ax2 = fig.add_subplot(gs[1, :])
ax2.fill_between(drawdown.reindex(common_idx).index, drawdown.reindex(common_idx) * 100, 0,
                 color="steelblue", alpha=0.4, label="Strategy")
ax2.fill_between(qqq_dd.index, qqq_dd * 100, 0,
                 color="darkorange", alpha=0.3, label="Nasdaq-100 (QQQ)")
ax2.set_title("Drawdown (%)")
ax2.set_ylabel("%")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Rolling annualised volatility (63-day)
ax3 = fig.add_subplot(gs[2, 0])
roll_vol = r.rolling(63).std() * np.sqrt(252) * 100
ax3.plot(roll_vol.reindex(common_idx), color="steelblue", linewidth=1.2, label="Strategy")
ax3.plot(qqq_roll_vol, color="darkorange", linewidth=1.2, linestyle="--", label="QQQ")
ax3.set_title("Rolling 63-day Ann. Vol (%)")
ax3.set_ylabel("%")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Rebalance turnover
ax4 = fig.add_subplot(gs[2, 1])
if len(result.turnover_history) > 0:
    ax4.bar(result.turnover_history.index, result.turnover_history * 100,
            width=15, color="slategray", alpha=0.7)
ax4.set_title("One-Way Turnover per Rebalance (%)")
ax4.set_ylabel("%")
ax4.grid(True, alpha=0.3)

# Annotation box with strategy vs QQQ stats
summary = result.summary()
qqq_r = qqq_log_ret.loc[common_idx]
qqq_ann_ret = float(np.exp(qqq_r.mean() * 252) - 1)
qqq_ann_vol = float(qqq_r.std() * np.sqrt(252))
qqq_sharpe = qqq_ann_ret / (qqq_ann_vol + 1e-9)
qqq_max_dd = float((qqq_dd_log - qqq_dd_log.cummax()).min())
ann = (
    f"{'Metric':<14} {'Strategy':>10} {'QQQ':>8}\n"
    f"{'-'*34}\n"
    f"{'Ann. Return':<14} {summary['annualised_return']*100:>9.1f}% {qqq_ann_ret*100:>7.1f}%\n"
    f"{'Ann. Vol':<14} {summary['annualised_vol']*100:>9.1f}% {qqq_ann_vol*100:>7.1f}%\n"
    f"{'Sharpe':<14} {summary['sharpe_ratio']:>10.2f} {qqq_sharpe:>8.2f}\n"
    f"{'Max DD':<14} {summary['max_drawdown']*100:>9.1f}% {qqq_max_dd*100:>7.1f}%"
)
fig.text(0.62, 0.97, ann, transform=fig.transFigure,
         fontsize=8.5, verticalalignment="top", family="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.suptitle("FactorAlpha Backtest — Momentum + Low Vol (Max Sharpe)", fontsize=13)
plt.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved to backtest_result.png")
