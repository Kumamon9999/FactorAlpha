"""Tests for Qlib ↔ FactorAlpha format conversion utilities."""

import numpy as np
import pandas as pd
import pytest

from factor_alpha.data.qlib_adapter import (
    to_qlib_multiindex,
    from_qlib_predictions,
    qlib_to_wide,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices():
    dates = pd.date_range("2023-01-02", periods=10, freq="B")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        100 * np.exp(rng.normal(0, 0.01, (10, 4)).cumsum(axis=0)),
        index=dates,
        columns=["AAA", "BBB", "CCC", "DDD"],
    )


@pytest.fixture
def sample_volumes(sample_prices):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        rng.integers(100_000, 1_000_000, sample_prices.shape).astype(float),
        index=sample_prices.index,
        columns=sample_prices.columns,
    )


# ---------------------------------------------------------------------------
# to_qlib_multiindex
# ---------------------------------------------------------------------------

def test_to_qlib_multiindex_basic_shape(sample_prices):
    qlib_df = to_qlib_multiindex(sample_prices)
    assert isinstance(qlib_df.index, pd.MultiIndex)
    assert qlib_df.index.names == ["datetime", "instrument"]
    assert "$close" in qlib_df.columns
    assert qlib_df.shape == (len(sample_prices) * len(sample_prices.columns), 1)


def test_to_qlib_multiindex_with_volume(sample_prices, sample_volumes):
    qlib_df = to_qlib_multiindex(sample_prices, volumes=sample_volumes)
    assert "$close" in qlib_df.columns
    assert "$volume" in qlib_df.columns
    assert qlib_df.shape[1] == 2


def test_to_qlib_multiindex_extra_fields(sample_prices):
    extra = sample_prices * 1.01  # fake open prices
    qlib_df = to_qlib_multiindex(sample_prices, open=extra)
    assert "$open" in qlib_df.columns
    assert "$close" in qlib_df.columns


def test_to_qlib_multiindex_dollar_prefix_not_doubled(sample_prices):
    """If caller passes a key that already starts with $, don't double the prefix."""
    qlib_df = to_qlib_multiindex(sample_prices, **{"$vwap": sample_prices * 0.99})
    assert "$vwap" in qlib_df.columns
    assert "$$vwap" not in qlib_df.columns


def test_to_qlib_multiindex_tickers_as_instruments(sample_prices):
    qlib_df = to_qlib_multiindex(sample_prices)
    instruments = qlib_df.index.get_level_values("instrument").unique().tolist()
    assert set(instruments) == set(sample_prices.columns)


def test_to_qlib_multiindex_dates_preserved(sample_prices):
    qlib_df = to_qlib_multiindex(sample_prices)
    datetimes = qlib_df.index.get_level_values("datetime").unique()
    assert len(datetimes) == len(sample_prices)


def test_to_qlib_multiindex_values_correct(sample_prices):
    qlib_df = to_qlib_multiindex(sample_prices)
    first_date = sample_prices.index[0]
    for ticker in sample_prices.columns:
        expected = sample_prices.loc[first_date, ticker]
        actual = qlib_df.loc[(first_date, ticker), "$close"]
        assert abs(actual - expected) < 1e-10


def test_to_qlib_multiindex_empty_raises():
    with pytest.raises(ValueError):
        to_qlib_multiindex(pd.DataFrame())


def test_to_qlib_multiindex_sorted(sample_prices):
    qlib_df = to_qlib_multiindex(sample_prices)
    assert qlib_df.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# from_qlib_predictions
# ---------------------------------------------------------------------------

@pytest.fixture
def qlib_predictions(sample_prices):
    """Fake Qlib prediction Series with (datetime, instrument) MultiIndex."""
    rng = np.random.default_rng(2)
    qlib_df = to_qlib_multiindex(sample_prices)
    scores = rng.normal(0, 1, len(qlib_df))
    return pd.Series(scores, index=qlib_df.index, name="score")


def test_from_qlib_predictions_shape(qlib_predictions, sample_prices):
    df = from_qlib_predictions(qlib_predictions)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (len(sample_prices), len(sample_prices.columns))


def test_from_qlib_predictions_columns_are_tickers(qlib_predictions, sample_prices):
    df = from_qlib_predictions(qlib_predictions)
    assert set(df.columns) == set(sample_prices.columns)


def test_from_qlib_predictions_index_is_datetime(qlib_predictions):
    df = from_qlib_predictions(qlib_predictions)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_from_qlib_predictions_reindex_dates(qlib_predictions, sample_prices):
    extra_dates = pd.date_range("2023-01-02", periods=20, freq="B")
    df = from_qlib_predictions(qlib_predictions, expected_dates=extra_dates)
    assert len(df) == 20
    # rows outside original range should be NaN
    assert df.iloc[-1].isna().all()


def test_from_qlib_predictions_reindex_tickers(qlib_predictions, sample_prices):
    subset = pd.Index(["AAA", "BBB", "ZZZ"])
    df = from_qlib_predictions(qlib_predictions, expected_tickers=subset)
    assert list(df.columns) == ["AAA", "BBB", "ZZZ"]
    assert df["ZZZ"].isna().all()  # ZZZ not in original data


def test_from_qlib_predictions_non_multiindex_raises():
    s = pd.Series([1.0, 2.0], index=pd.Index(["A", "B"]))
    with pytest.raises(ValueError):
        from_qlib_predictions(s)


def test_round_trip(sample_prices):
    """to_qlib_multiindex → from_qlib_predictions should recover original close values."""
    qlib_df = to_qlib_multiindex(sample_prices)
    pred = qlib_df["$close"]  # use close as fake predictions
    recovered = from_qlib_predictions(pred)
    pd.testing.assert_frame_equal(
        recovered.sort_index(axis=1),
        sample_prices.sort_index(axis=1),
        check_names=False,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# qlib_to_wide
# ---------------------------------------------------------------------------

def test_qlib_to_wide_shape(sample_prices):
    qlib_df = to_qlib_multiindex(sample_prices, volumes=sample_prices * 1000)
    wide = qlib_to_wide(qlib_df, field="$close")
    assert wide.shape == sample_prices.shape


def test_qlib_to_wide_values(sample_prices):
    qlib_df = to_qlib_multiindex(sample_prices)
    wide = qlib_to_wide(qlib_df, "$close")
    pd.testing.assert_frame_equal(
        wide.sort_index(axis=1),
        sample_prices.sort_index(axis=1),
        check_names=False,
        atol=1e-10,
    )


def test_qlib_to_wide_unknown_field_raises(sample_prices):
    qlib_df = to_qlib_multiindex(sample_prices)
    with pytest.raises(KeyError):
        qlib_to_wide(qlib_df, "$nonexistent")
