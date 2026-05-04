# FactorAlpha

A personal quantitative research and portfolio construction framework, built end-to-end from raw market data to backtested strategy returns.

---

## Background

This started as a final project, but it became something bigger than that.

The core idea was to build a system that covers the full quant workflow — data, factors, risk, optimization, and backtest — rather than assembling disconnected scripts. I sketched the basic architecture and used Claude Code to help translate the design into working Python: each module has a clear interface, tests are first-class, and the pipeline is built to be extended, not thrown away after a grade.

Halfway through, I realized I was only thinking in daily data. After talking with alumni and friends working at quant funds and asset managers, I learned that most production systems don't treat intraday and daily signals as separate worlds — they combine them into a unified signal and execution layer. That conversation led to the `multi-frequency` branch of the system: daily factor signals drive portfolio construction, while 5-minute bars drive execution simulation and intraday features.

This project covers: data engineering, factor research, machine learning, risk modeling, convex optimization, parallel computing, backtesting, and software engineering practices (typed APIs, pytest, git). That intersection is exactly what I think makes a strong final project — and a useful foundation for real work.

I will not stop here. This repository is my personal quant lab going forward.

---

## Architecture

```
FactorAlpha/
├── src/factor_alpha/
│   ├── data/           # Market data ingestion and preprocessing
│   ├── factors/        # Alpha signal generation (traditional + ML)
│   ├── analysis/       # Factor research and evaluation
│   ├── risk/           # Barra-style factor risk model
│   ├── optimization/   # Portfolio construction via CVXPY
│   └── backtest/       # Walk-forward engine + intraday execution
└── tests/              # 111 pytest tests across all modules
```

The pipeline flows in one direction:

```
Raw Data → Returns / Prices
    → Factor Scores (traditional or ML)
        → Cross-sectional Z-score + Composite
            → Factor Risk Model (covariance)
                → Portfolio Optimizer (weights)
                    → Backtest Engine (performance)
                        → Intraday Execution Simulator (realistic fills)
```

---

## Modules

### `data/` — Market Data

**`loader.py`**

Three loaders share a common `_PriceLoaderMixin` (`.get_returns()`, `.align()`):

| Class | Source |
|---|---|
| `UniverseLoader` | Yahoo Finance via `yfinance` |
| `CSVLoader` | Local CSV, wide or long format |
| `AShareZipLoader` | Per-day ZIP archives of A-share 5-min bars, resampled to daily OHLCV |

All loaders validate data quality (drops tickers with >20% missing), forward/backward fill gaps, and expose `.prices`, `.volumes`, `.get_returns()`, `.get_dollar_volume()`, `.align()`.

**`intraday.py`**

- `AShareIntradayZipLoader`: loads raw 5-minute bars without collapsing to daily.
- `IntradayFeatureBuilder`: converts intraday bars to daily factor-ready matrices — VWAP, realized volatility, close-to-VWAP gap, intraday momentum, morning return.
- `BarSpec` / `ASHARE_5MIN_SPEC`: metadata container describing bar frequency and annualization constants.

---

### `factors/` — Alpha Signal Generation

All factors extend `BaseFactor` and produce a `(dates × tickers)` DataFrame of raw scores.

**Traditional factors** (`factors/traditional/`)

| Class | Signal |
|---|---|
| `MomentumFactor` | 12-1 momentum: cumulative log return over `[lookback, skip]` window |
| `ShortTermReversalFactor` | Negative of recent cumulative return (mean-reversion) |
| `LowVolatilityFactor` | Negative annualized realized volatility |
| `IdiosyncraticVolatilityFactor` | Negative idiosyncratic volatility after removing market beta |

**ML factors** (`factors/ml/`)

| Class | Method |
|---|---|
| `PCAResidualFactor` | PCA reconstruction residuals — persistent idiosyncratic alpha vs. systematic components |
| `RidgeAlphaFactor` | Cross-sectional Ridge regression on lagged returns and vol-regime features |
| `GradientBoostFactor` | Gradient Boosted Trees — captures non-linear momentum / vol-regime interactions |
| `RandomForestFactor` | Random Forest ensemble — exposes `feature_importances_` after fitting |

All ML factors support two modes automatically: rolling expanding-window (for `FactorResearch` IC evaluation) and single-date prediction (for `BacktestEngine`).

`GradientBoostFactor` and `RandomForestFactor` share their walk-forward `compute()` logic through a `_TreeFactor` base class — subclasses only override `_make_model()`.

**Cross-sectional utilities** (`factors/cross_section.py`)

- `cross_sectional_zscore()`: row-wise standardization with winsorization
- `rank_normalize()`: percentile ranks scaled to `[-1, 1]`
- `composite_score()`: weighted combination of multiple factor DataFrames
- `neutralize()`: sector/group-level demeaning

**`PrecomputedFactor`**: wraps any pre-computed score matrix (e.g. intraday features) so it can be dropped into the pipeline like any other factor.

---

### `analysis/` — Factor Research

`FactorResearch` evaluates the predictive quality of a single factor:

| Method | What it measures |
|---|---|
| `.ic(forward_period, method)` | Daily cross-sectional rank correlation with forward returns |
| `.icir(forward_period)` | IC / std(IC) — signal-to-noise ratio |
| `.decay_profile(max_horizon)` | Mean IC across horizons 1…N — how fast the signal decays. Parallelized over horizons via `ThreadPoolExecutor`. |
| `.quintile_returns(forward_periods)` | Annualized mean return per score quintile |
| `.turnover(top_pct)` | Daily turnover of the long book |
| `.evaluate(name)` | Runs all diagnostics and returns a `FactorStats` dataclass |

---

### `risk/` — Factor Risk Model

`FactorRiskModel` implements a Barra-style linear factor model:

```
r = B @ f + ε
```

where `r` is `(T × N)` asset returns, `B` is `(N × K)` factor loadings (OLS), `f` is `(T × K)` factor returns, and `ε` is idiosyncratic return.

Portfolio covariance:

```
Σ_p = B Σ_f B' + D
```

| Method | Output |
|---|---|
| `.fit()` | Estimates B, Σ_f, and idiosyncratic variances D via OLS |
| `.total_covariance()` | Full N×N covariance matrix |
| `.portfolio_variance(weights)` | Annualized portfolio variance |
| `.attribute(weights)` | Full risk decomposition: factor vol, idio vol, parametric VaR/CVaR at 95% and 99% |
| `.historical_var(weights, confidence)` | Empirical VaR from fitted return history |

Factor return series are constructed from cross-sectional score DataFrames by rank-normalizing each row into a long-short portfolio.

---

### `optimization/` — Portfolio Construction

`PortfolioOptimizer` builds target weights from factor scores and a covariance matrix using [CVXPY](https://www.cvxpy.org/) with the CLARABEL solver.

| Method | Objective |
|---|---|
| `max_sharpe()` | Parametric frontier sweep, picks portfolio with highest empirical Sharpe. Supports turnover penalty. |
| `min_variance()` | Minimum portfolio variance subject to optional factor exposure floor |
| `risk_parity()` | Equal marginal risk contributions via iterative Newton method |
| `factor_tilt()` | Maximize factor score subject to a portfolio volatility budget |

All methods support `long_only`, `max_weight`, and fall back to equal-weight if the solver fails.

---

### `backtest/` — Strategy Evaluation

**`BacktestEngine`** (`engine.py`)

Walk-forward simulation that rebalances at each date in a monthly, weekly, or every-N-day schedule:

1. Slice a trailing window of prices and returns
2. Optionally apply a `universe_selector` callable (e.g. top-N by dollar volume)
3. Compute all factors **in parallel** via `ThreadPoolExecutor` — each `factor.compute()` only reads its window, so concurrent execution is safe
4. Z-score and combine into a composite signal
5. Fit the risk model and compute the covariance matrix
6. Run the chosen optimizer
7. Record portfolio returns, turnover, risk attribution, and factor exposures

Returns a `BacktestResult` with cumulative return, annualized return/vol, Sharpe, max drawdown, Calmar ratio, and full history DataFrames.

**`IntradayExecutionSimulator`** (`execution.py`)

Takes daily target weights from `BacktestEngine` and 5-minute bar data to simulate realistic execution:

- Fill prices at VWAP, open, or close
- Transaction costs in basis points
- One-way turnover tracking per rebalance

Returns an `ExecutionResult` with realized portfolio returns net of costs, trade cost series, turnover history, and fill prices.

---

## Multi-Frequency Design

The system is designed around one insight from industry conversations: **daily and intraday data should not be separate silos.**

The intended flow:
1. Daily factors (`MomentumFactor`, ML factors) generate portfolio-level signals
2. `IntradayFeatureBuilder` extracts intraday signals (VWAP gap, realized vol, morning momentum) from 5-min bars and wraps them as `PrecomputedFactor` objects
3. Both feed into the same `composite_score()` → `BacktestEngine` pipeline
4. `IntradayExecutionSimulator` converts daily target weights into 5-min fills with realistic costs

This mirrors how multi-frequency quant systems actually work in practice: signals at different horizons, unified into one weight.

---

## Tests

```bash
pytest tests/ -q
# 111 passed
```

| Test file | Coverage |
|---|---|
| `test_data_loader.py` | `UniverseLoader`, `CSVLoader` (wide + long), date clipping, dollar volume, error paths |
| `test_factors.py` | All traditional + ML factors, cross-sectional transforms, `neutralize`, `PrecomputedFactor` |
| `test_analysis.py` | IC range, ICIR sign, decay profile, quintile ordering, turnover, `FactorStats` |
| `test_risk.py` | Single- and multi-factor models, PSD covariance, VaR/CVaR ordering, risk attribution identity |
| `test_optimization.py` | All four optimizer modes, weight constraints, risk budget, equal-risk contributions |
| `test_backtest.py` | Monthly/weekly/integer rebalance, universe selector, factor-tilt mode, risk history |
| `test_intraday.py` | OHLCV aggregation, intraday feature matrices, execution simulator fills and costs |

---

## Installation

```bash
pip install -e ".[dev]"
```

Dependencies: `pandas`, `numpy`, `yfinance`, `scipy`, `cvxpy`, `scikit-learn`, `statsmodels`.  
Dev: `pytest`, `pytest-cov`.

---

## What's Next

This is a living repository. Planned additions:

- **Live data**: replace `yfinance` with a broker API (Alpaca / Interactive Brokers)
- **Alternative data**: news sentiment, options flow as `PrecomputedFactor` inputs
- **Online learning**: incremental ML factor updates without full refit
- **Transaction cost model**: market impact beyond flat bps
- **Portfolio monitoring**: real-time risk attribution dashboard
- **Chinese A-share universe expansion**: full 4000+ stock coverage with sector neutralization
