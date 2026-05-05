"""
Bidirectional format conversion between FactorAlpha and Qlib.

FactorAlpha uses (dates × tickers) wide DataFrames throughout.
Qlib uses a (datetime, instrument) MultiIndex DataFrame.

Use to_qlib_multiindex() before passing data to Qlib APIs.
Use from_qlib_predictions() to turn Qlib model outputs back into
a FactorAlpha-compatible (dates × tickers) score DataFrame.
"""

from typing import Optional

import numpy as np
import pandas as pd


def to_qlib_multiindex(
    prices: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    **extra: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert FactorAlpha wide DataFrames to Qlib's (datetime, instrument) MultiIndex.

    Parameters
    ----------
    prices : dates × tickers close prices  → stored as "$close"
    volumes : dates × tickers volumes (optional) → stored as "$volume"
    **extra : any additional wide DataFrames, e.g. open=df_open → stored as "$open"

    Returns
    -------
    DataFrame with MultiIndex (datetime, instrument) and columns [$close, ...]
    """
    if prices.empty:
        raise ValueError("prices must be a non-empty DataFrame")

    frames = {"$close": prices}
    if volumes is not None:
        frames["$volume"] = volumes
    for name, df in extra.items():
        key = name if name.startswith("$") else f"${name}"
        frames[key] = df

    long: list[pd.Series] = []
    for col_name, df in frames.items():
        s = df.stack(future_stack=True)
        s.name = col_name
        s.index.names = ["datetime", "instrument"]
        long.append(s)

    result = pd.concat(long, axis=1).sort_index()
    result.index.names = ["datetime", "instrument"]
    return result


def from_qlib_predictions(
    predictions: pd.Series,
    expected_dates: Optional[pd.DatetimeIndex] = None,
    expected_tickers: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """
    Convert a Qlib prediction Series with (datetime, instrument) MultiIndex
    to a FactorAlpha (dates × tickers) DataFrame.

    Parameters
    ----------
    predictions : pd.Series with (datetime, instrument) MultiIndex
    expected_dates : optional DatetimeIndex to reindex rows
    expected_tickers : optional Index to reindex columns

    Returns
    -------
    dates × tickers DataFrame of scores
    """
    if not isinstance(predictions.index, pd.MultiIndex):
        raise ValueError("predictions must have a (datetime, instrument) MultiIndex")

    df = predictions.unstack(level="instrument")
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)

    if expected_dates is not None:
        df = df.reindex(index=expected_dates)
    if expected_tickers is not None:
        df = df.reindex(columns=expected_tickers)

    return df


def qlib_to_wide(
    qlib_df: pd.DataFrame,
    field: str = "$close",
) -> pd.DataFrame:
    """
    Extract a single field from a Qlib MultiIndex DataFrame as a wide (dates × tickers) frame.

    Parameters
    ----------
    qlib_df : DataFrame with (datetime, instrument) MultiIndex
    field : column name to extract, e.g. "$close"
    """
    if field not in qlib_df.columns:
        raise KeyError(f"Field '{field}' not found. Available: {list(qlib_df.columns)}")

    df = qlib_df[field].unstack(level="instrument")
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)
    return df
