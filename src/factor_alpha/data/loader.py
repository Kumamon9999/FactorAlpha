import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class UniverseLoader:
    """Fetches and preprocesses OHLCV data for a stock universe."""

    def __init__(self, tickers: List[str]):
        if not tickers:
            raise ValueError("tickers must be a non-empty list")
        self.tickers = tickers
        self._prices: Optional[pd.DataFrame] = None
        self._volumes: Optional[pd.DataFrame] = None

    def fetch(self, start: str, end: str) -> "UniverseLoader":
        """Download adjusted close prices and volumes from Yahoo Finance."""
        logger.info("Fetching %d tickers from %s to %s", len(self.tickers), start, end)
        raw = yf.download(self.tickers, start=start, end=end, auto_adjust=True, progress=False)

        if isinstance(raw.columns, pd.MultiIndex):
            self._prices = raw["Close"].copy()
            self._volumes = raw["Volume"].copy()
        else:
            # single ticker returned flat DataFrame
            self._prices = raw[["Close"]].rename(columns={"Close": self.tickers[0]})
            self._volumes = raw[["Volume"]].rename(columns={"Volume": self.tickers[0]})

        self._prices = self._prices.ffill().bfill()
        self._volumes = self._volumes.ffill().bfill()

        # drop tickers with >20% missing before fill
        threshold = 0.8
        valid = self._prices.notna().mean() >= threshold
        dropped = valid[~valid].index.tolist()
        if dropped:
            logger.warning("Dropping tickers with insufficient data: %s", dropped)
        self._prices = self._prices.loc[:, valid]
        self._volumes = self._volumes.loc[:, valid]

        logger.info("Universe: %d tickers, %d trading days", self._prices.shape[1], len(self._prices))
        return self

    @property
    def prices(self) -> pd.DataFrame:
        self._check_fetched()
        return self._prices

    @property
    def volumes(self) -> pd.DataFrame:
        self._check_fetched()
        return self._volumes

    def get_returns(self, method: str = "log") -> pd.DataFrame:
        """Compute daily returns. method: 'log' or 'simple'."""
        self._check_fetched()
        if method == "log":
            return np.log(self._prices / self._prices.shift(1)).iloc[1:]
        elif method == "simple":
            return self._prices.pct_change().iloc[1:]
        raise ValueError(f"method must be 'log' or 'simple', got '{method}'")

    def get_dollar_volume(self) -> pd.DataFrame:
        """Price × volume — proxy for liquidity."""
        self._check_fetched()
        return (self._prices * self._volumes).ffill()

    def align(self, *others: pd.DataFrame) -> List[pd.DataFrame]:
        """Align multiple DataFrames to the intersection of dates and tickers."""
        frames = [self._prices] + list(others)
        common_idx = frames[0].index
        common_cols = frames[0].columns
        for f in frames[1:]:
            common_idx = common_idx.intersection(f.index)
            common_cols = common_cols.intersection(f.columns)
        return [f.loc[common_idx, common_cols] for f in frames]

    def _check_fetched(self):
        if self._prices is None:
            raise RuntimeError("Call .fetch(start, end) before accessing data.")
