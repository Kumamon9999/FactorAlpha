"""Tests for UniverseLoader and CSVLoader."""

import io
import textwrap
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from factor_alpha.data.loader import UniverseLoader, CSVLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# UniverseLoader
# ---------------------------------------------------------------------------

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


def test_simple_returns_shape(mock_prices):
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


def test_not_fetched_get_returns_raises():
    loader = UniverseLoader(["AAPL"])
    with pytest.raises(RuntimeError):
        loader.get_returns()


def test_dollar_volume_shape(mock_prices):
    with patch("factor_alpha.data.loader.yf.download", return_value=_make_yf_download(mock_prices)):
        loader = UniverseLoader(["AAPL", "MSFT", "GOOG"])
        loader.fetch("2023-01-01", "2023-06-01")
        dv = loader.get_dollar_volume()
        assert dv.shape == mock_prices.shape


def test_align_intersects_index(mock_prices):
    with patch("factor_alpha.data.loader.yf.download", return_value=_make_yf_download(mock_prices)):
        loader = UniverseLoader(["AAPL", "MSFT", "GOOG"])
        loader.fetch("2023-01-01", "2023-06-01")
        other = mock_prices.iloc[10:50, :2]  # shorter date range, fewer tickers
        aligned = loader.align(other)
        assert len(aligned) == 2
        assert len(aligned[0]) == len(aligned[1])
        assert list(aligned[0].columns) == list(aligned[1].columns)


# ---------------------------------------------------------------------------
# CSVLoader — wide format
# ---------------------------------------------------------------------------

@pytest.fixture
def wide_csv(tmp_path):
    csv = textwrap.dedent("""\
        date,AAPL,MSFT,GOOG
        2023-01-02,150.0,250.0,100.0
        2023-01-03,152.0,252.0,101.0
        2023-01-04,151.0,251.0,102.0
        2023-01-05,153.0,253.0,103.0
        2023-01-06,154.0,254.0,104.0
    """)
    p = tmp_path / "prices.csv"
    p.write_text(csv)
    return str(p)


def test_csv_loader_wide_prices_shape(wide_csv):
    loader = CSVLoader(wide_csv, fmt="wide")
    assert loader.prices.shape == (5, 3)
    assert set(loader.prices.columns) == {"AAPL", "MSFT", "GOOG"}


def test_csv_loader_wide_index_is_datetime(wide_csv):
    loader = CSVLoader(wide_csv, fmt="wide")
    assert isinstance(loader.prices.index, pd.DatetimeIndex)


def test_csv_loader_wide_get_returns(wide_csv):
    loader = CSVLoader(wide_csv, fmt="wide")
    ret = loader.get_returns("log")
    assert ret.shape == (4, 3)
    assert not ret.isna().any().any()


def test_csv_loader_wide_simple_returns(wide_csv):
    loader = CSVLoader(wide_csv, fmt="wide")
    ret = loader.get_returns("simple")
    assert ret.shape == (4, 3)


def test_csv_loader_invalid_method_raises(wide_csv):
    loader = CSVLoader(wide_csv, fmt="wide")
    with pytest.raises(ValueError):
        loader.get_returns("bad")


def test_csv_loader_date_clip(wide_csv):
    loader = CSVLoader(wide_csv, fmt="wide", start="2023-01-03", end="2023-01-05")
    assert len(loader.prices) == 3


def test_csv_loader_align(wide_csv):
    loader = CSVLoader(wide_csv, fmt="wide")
    other = loader.prices.iloc[1:3, :2]
    aligned = loader.align(other)
    assert len(aligned[0]) == 2
    assert len(aligned[1]) == 2
    assert aligned[0].columns.tolist() == aligned[1].columns.tolist()


def test_csv_loader_volumes_dollar_volume(tmp_path):
    prices_csv = textwrap.dedent("""\
        date,AAPL,MSFT
        2023-01-02,150.0,250.0
        2023-01-03,152.0,252.0
        2023-01-04,151.0,251.0
    """)
    volumes_csv = textwrap.dedent("""\
        date,AAPL,MSFT
        2023-01-02,1000,2000
        2023-01-03,1100,2100
        2023-01-04,1050,2050
    """)
    pp = tmp_path / "prices.csv"
    vp = tmp_path / "volumes.csv"
    pp.write_text(prices_csv)
    vp.write_text(volumes_csv)

    loader = CSVLoader(str(pp), volumes_path=str(vp), fmt="wide")
    dv = loader.get_dollar_volume()
    assert dv.shape == (3, 2)
    assert (dv > 0).all().all()


def test_csv_loader_no_volumes_raises(wide_csv):
    loader = CSVLoader(wide_csv, fmt="wide")
    with pytest.raises(RuntimeError):
        loader.get_dollar_volume()


# ---------------------------------------------------------------------------
# CSVLoader — long format
# ---------------------------------------------------------------------------

@pytest.fixture
def long_csv(tmp_path):
    csv = textwrap.dedent("""\
        date,ticker,close
        2023-01-02,AAPL,150.0
        2023-01-02,MSFT,250.0
        2023-01-03,AAPL,152.0
        2023-01-03,MSFT,252.0
        2023-01-04,AAPL,151.0
        2023-01-04,MSFT,251.0
    """)
    p = tmp_path / "long_prices.csv"
    p.write_text(csv)
    return str(p)


def test_csv_loader_long_shape(long_csv):
    loader = CSVLoader(long_csv, fmt="long", price_col="close")
    assert loader.prices.shape == (3, 2)
    assert set(loader.prices.columns) == {"AAPL", "MSFT"}


def test_csv_loader_long_returns(long_csv):
    loader = CSVLoader(long_csv, fmt="long", price_col="close")
    ret = loader.get_returns()
    assert ret.shape == (2, 2)


def test_csv_loader_invalid_fmt_raises(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("date,A\n2023-01-02,1.0\n")
    with pytest.raises(ValueError):
        CSVLoader(str(p), fmt="diagonal")
