import numpy as np
import pandas as pd

from ..base import BaseFactor


class LowVolatilityFactor(BaseFactor):
    """
    Negative annualised realised volatility.
    Lower-vol stocks score higher (long low-vol, short high-vol).
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"low_vol_{self.lookback}"

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        ann_vol = returns.rolling(self.lookback).std() * np.sqrt(252)
        return -ann_vol


class IdiosyncraticVolatilityFactor(BaseFactor):
    """
    Negative idiosyncratic volatility: residual vol after removing
    market (equal-weight) return. Low idio-vol stocks tend to outperform.
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"idio_vol_{self.lookback}"

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        market_ret = returns.mean(axis=1)
        beta = returns.rolling(self.lookback).cov(market_ret).div(
            market_ret.rolling(self.lookback).var()
        )
        systematic_vol = beta.abs() * market_ret.rolling(self.lookback).std() * np.sqrt(252)
        total_vol = returns.rolling(self.lookback).std() * np.sqrt(252)
        idio = (total_vol**2 - systematic_vol**2).clip(lower=0) ** 0.5
        return -idio
