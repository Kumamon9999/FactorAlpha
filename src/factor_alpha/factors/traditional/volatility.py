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
        mkt_var = market_ret.rolling(self.lookback).var().replace(0, np.nan)

        # DataFrame.rolling().cov(Series) produces a MultiIndex output in pandas 2.x,
        # so we use the identity cov(X,Y) = E[XY] - E[X]*E[Y] instead.
        cov = (
            returns.multiply(market_ret, axis=0).rolling(self.lookback).mean()
            - returns.rolling(self.lookback).mean().multiply(
                market_ret.rolling(self.lookback).mean(), axis=0
            )
        )
        beta = cov.div(mkt_var, axis=0)

        mkt_vol = market_ret.rolling(self.lookback).std() * np.sqrt(252)
        systematic_var = (beta ** 2).multiply(mkt_vol ** 2, axis=0)
        total_var = (returns.rolling(self.lookback).std() * np.sqrt(252)) ** 2
        idio = (total_var - systematic_var).clip(lower=0) ** 0.5
        return -idio
