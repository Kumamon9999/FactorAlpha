"""
Multi-frequency FactorAlpha workflow.

The signal and portfolio clocks stay daily/monthly, while the data and
execution layers can use 5-minute bars.
"""

import os

import numpy as np

from factor_alpha import (
    AShareIntradayZipLoader,
    BacktestEngine,
    IntradayExecutionSimulator,
    IntradayFeatureBuilder,
    LowVolatilityFactor,
    MomentumFactor,
    PrecomputedFactor,
)


DATASET_DIR = os.path.join(os.path.dirname(__file__), "src/factor_alpha/data/dataset")
TICKERS = ["sh600519", "sz000858", "sh600036", "sz000001", "sz300750"]


loader = AShareIntradayZipLoader(
    dataset_dir=DATASET_DIR,
    tickers=TICKERS,
    start="2025-01-01",
    end="2025-03-31",
)
bars = loader.load()

features = IntradayFeatureBuilder(bars)
daily = features.daily_ohlcv()

prices = daily["close"]
returns = np.log(prices / prices.shift(1)).dropna()

intraday_vol = PrecomputedFactor(
    "intraday_realized_vol",
    -features.intraday_realized_vol(),
)
close_vwap_gap = PrecomputedFactor(
    "close_to_vwap_gap",
    features.close_to_vwap_gap(),
)

engine = BacktestEngine(
    prices=prices,
    returns=returns,
    factors=[
        MomentumFactor(lookback=20, skip=5),
        LowVolatilityFactor(lookback=20),
        intraday_vol,
        close_vwap_gap,
    ],
    lookback=30,
    rebalance_freq="monthly",
    optimizer_mode="min_variance",
    max_weight=0.40,
)
daily_result = engine.run()

execution = IntradayExecutionSimulator(bars, fill="vwap", cost_bps=5)
executed_result = execution.simulate(daily_result.weights_history)

print("Daily signal backtest:")
print(daily_result.summary())
print("\n5-minute execution replay:")
print({
    "cumulative_return": round(executed_result.cumulative_return, 4),
    "mean_turnover": round(float(executed_result.turnover_history.mean()), 4),
    "total_cost": round(float(executed_result.trade_costs.sum()), 4),
})
