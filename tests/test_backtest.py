"""Tests for BacktestEngine end-to-end pipeline."""

import numpy as np
import pandas as pd
import pytest

from factor_alpha.backtest.engine import BacktestEngine, BacktestResult
from factor_alpha.factors import MomentumFactor, LowVolatilityFactor


def test_backtest_runs_and_returns_result(returns, prices):
    factors = [MomentumFactor(lookback=60, skip=5), LowVolatilityFactor(lookback=30)]
    engine = BacktestEngine(
        prices=prices,
        returns=returns,
        factors=factors,
        lookback=120,
        rebalance_freq="monthly",
        optimizer_mode="min_variance",
        max_weight=0.15,
    )
    result = engine.run()
    assert isinstance(result, BacktestResult)


def test_backtest_portfolio_returns_length(returns, prices):
    factors = [MomentumFactor(lookback=60, skip=5)]
    engine = BacktestEngine(prices=prices, returns=returns, factors=factors, lookback=120)
    result = engine.run()
    assert len(result.portfolio_returns) == len(returns)


def test_backtest_weights_sum_to_one(returns, prices):
    factors = [LowVolatilityFactor(lookback=30)]
    engine = BacktestEngine(
        prices=prices, returns=returns, factors=factors,
        lookback=100, optimizer_mode="risk_parity"
    )
    result = engine.run()
    if len(result.weights_history) > 0:
        row_sums = result.weights_history.sum(axis=1)
        assert (row_sums.abs() < 1e-3).all() or (abs(row_sums - 1.0) < 1e-3).all()


def test_backtest_summary_keys(returns, prices):
    factors = [MomentumFactor(lookback=60, skip=5)]
    engine = BacktestEngine(prices=prices, returns=returns, factors=factors, lookback=120)
    result = engine.run()
    summary = result.summary()
    for key in ("annualised_return", "annualised_vol", "sharpe_ratio", "max_drawdown"):
        assert key in summary


def test_backtest_turnover_in_0_1(returns, prices):
    factors = [MomentumFactor(lookback=60, skip=5)]
    engine = BacktestEngine(prices=prices, returns=returns, factors=factors, lookback=120)
    result = engine.run()
    if len(result.turnover_history) > 0:
        assert (result.turnover_history >= 0).all()
        assert (result.turnover_history <= 1.01).all()


def test_backtest_weekly_rebalance(returns, prices):
    factors = [LowVolatilityFactor(lookback=30)]
    engine = BacktestEngine(
        prices=prices, returns=returns, factors=factors,
        lookback=80, rebalance_freq="weekly",
    )
    result = engine.run()
    assert isinstance(result, BacktestResult)


def test_backtest_integer_rebalance(returns, prices):
    """Rebalance every 21 trading days."""
    factors = [MomentumFactor(lookback=60, skip=5)]
    engine = BacktestEngine(
        prices=prices, returns=returns, factors=factors,
        lookback=100, rebalance_freq=21,
    )
    result = engine.run()
    assert isinstance(result, BacktestResult)


def test_backtest_invalid_rebalance_raises(returns, prices):
    factors = [MomentumFactor(lookback=60, skip=5)]
    engine = BacktestEngine(
        prices=prices, returns=returns, factors=factors,
        lookback=100, rebalance_freq="quarterly",
    )
    with pytest.raises(ValueError):
        engine.run()


def test_backtest_universe_selector(returns, prices):
    """Universe selector should narrow down tickers."""
    factors = [LowVolatilityFactor(lookback=30)]

    def top_half(date, price_win, ret_win):
        """Keep only the first half of tickers alphabetically."""
        return sorted(price_win.columns.tolist())[: len(price_win.columns) // 2]

    engine = BacktestEngine(
        prices=prices, returns=returns, factors=factors,
        lookback=80, rebalance_freq="monthly",
        universe_selector=top_half,
    )
    result = engine.run()
    assert isinstance(result, BacktestResult)
    if len(result.weights_history) > 0:
        # universe was restricted, so max tickers held ≤ half of original
        max_active = (result.weights_history > 1e-6).sum(axis=1).max()
        assert max_active <= prices.shape[1] // 2 + 1


def test_backtest_factor_tilt_mode(returns, prices):
    factors = [MomentumFactor(lookback=60, skip=5)]
    engine = BacktestEngine(
        prices=prices, returns=returns, factors=factors,
        lookback=100, optimizer_mode="factor_tilt", risk_budget=0.20,
    )
    result = engine.run()
    assert isinstance(result, BacktestResult)


def test_backtest_summary_calmar_present(returns, prices):
    factors = [MomentumFactor(lookback=60, skip=5)]
    engine = BacktestEngine(prices=prices, returns=returns, factors=factors, lookback=120)
    result = engine.run()
    assert "calmar_ratio" in result.summary()


def test_backtest_risk_history_populated(returns, prices):
    factors = [LowVolatilityFactor(lookback=30)]
    engine = BacktestEngine(
        prices=prices, returns=returns, factors=factors,
        lookback=80, rebalance_freq="monthly",
    )
    result = engine.run()
    assert len(result.risk_history) > 0
    assert "total_vol" in result.risk_history.columns
