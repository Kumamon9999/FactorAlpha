"""Tests for UniverseLoader (Yahoo Finance calls are mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from factor_alpha.data.loader import UniverseLoader


@pytest.fixture
def mock_prices():
    dates = pd.date_range("2023-01-02", periods=100, freq="B")
    rng = np.random.default_rng(0)
    data = 100 * np.exp(rng.normal(0, 0.01, (100, 3)).cumsum(axis=0))
    return pd.DataFrame(data, index=dates, columns=["AAPL", "MSFT", "GOOG"])


def _make_yf_download(mock_prices):
    """Return a fake yfinance download result (MultiIndex columns)."""
    cols = pd.MultiIndex.from_tuples(
        [("Close", t) for t in mock_prices.columns]
        + [("Volume", t) for t in mock_prices.columns]
    )
    vol = pd.DataFrame(
        1_000_000 * np.ones((len(mock_prices), len(mock_prices.columns))),
        index=mock_prices.index,
        columns=mock_prices.columns,
    )
    combined = pd.concat([mock_prices, vol], axis=1)
    combined.columns = cols
    return combined


def test_fetch_returns_prices(mock_prices):
    with patch("factor_alpha.data.loader.yf.download", return_value=_make_yf_download(mock_prices)):
        loader = UniverseLoader(["AAPL", "MSFT", "GOOG"])
        loader.fetch("2023-01-01", "2023-06-01")
        assert set(loader.prices.columns) == {"AAPL", "MSFT", "GOOG"}
        assert len(loader.prices) == 100


def test_log_returns_shape(mock_prices):
    with patch("factor_alpha.data.loader.yf.download", return_value=_make_yf_download(mock_prices)):
        loader = UniverseLoader(["AAPL", "MSFT", "GOOG"])
        loader.fetch("2023-01-01", "2023-06-01")
        ret = loader.get_returns("log")
        assert ret.shape == (99, 3)
        assert not ret.isna().any().any()


def test_simple_returns_positive_on_rising_prices(mock_prices):
    with patch("factor_alpha.data.loader.yf.download", return_value=_make_yf_download(mock_prices)):
        loader = UniverseLoader(["AAPL", "MSFT", "GOOG"])
        loader.fetch("2023-01-01", "2023-06-01")
        simple = loader.get_returns("simple")
        assert simple.shape[1] == 3


def test_invalid_return_type(mock_prices):
    with patch("factor_alpha.data.loader.yf.download", return_value=_make_yf_download(mock_prices)):
        loader = UniverseLoader(["AAPL", "MSFT", "GOOG"])
        loader.fetch("2023-01-01", "2023-06-01")
        with pytest.raises(ValueError):
            loader.get_returns("invalid")


def test_empty_tickers_raises():
    with pytest.raises(ValueError):
        UniverseLoader([])


def test_not_fetched_raises():
    loader = UniverseLoader(["AAPL"])
    with pytest.raises(RuntimeError):
        _ = loader.prices
