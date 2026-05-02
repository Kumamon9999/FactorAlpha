"""Shared fixtures for FactorAlpha tests."""

import numpy as np
import pandas as pd
import pytest

RNG = np.random.default_rng(42)
TICKERS = [f"T{i:02d}" for i in range(20)]
DATES = pd.date_range("2020-01-02", periods=500, freq="B")


@pytest.fixture
def returns():
    """500-day log-return DataFrame for 20 synthetic tickers."""
    data = RNG.normal(0, 0.01, size=(len(DATES), len(TICKERS)))
    return pd.DataFrame(data, index=DATES, columns=TICKERS)


@pytest.fixture
def prices(returns):
    """Synthetic price series constructed from cumulative returns."""
    log_prices = returns.cumsum()
    return 100 * np.exp(log_prices)


@pytest.fixture
def factor_scores(returns):
    """Simple synthetic factor: 5-day rolling mean return (momentum proxy)."""
    raw = returns.rolling(5).mean()
    mu = raw.mean(axis=1)
    sigma = raw.std(axis=1).replace(0, 1)
    return raw.subtract(mu, axis=0).divide(sigma, axis=0).fillna(0)
