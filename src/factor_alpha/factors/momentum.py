import pandas as pd

from .base import BaseFactor


class MomentumFactor(BaseFactor):
    """
    Classic 12-1 momentum: cumulative return over [lookback, skip] trading days.
    Skipping the most recent `skip` days avoids short-term reversal contamination.
    """

    def __init__(self, lookback: int = 252, skip: int = 21):
        if lookback <= skip:
            raise ValueError("lookback must be greater than skip")
        self.lookback = lookback
        self.skip = skip

    @property
    def name(self) -> str:
        return f"momentum_{self.lookback}_{self.skip}"

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        # cumulative return from t-lookback to t-skip
        log_cum = returns.rolling(self.lookback).sum() - returns.rolling(self.skip).sum()
        return log_cum


class ShortTermReversalFactor(BaseFactor):
    """
    1-week (5-day) reversal: negative of recent cumulative return.
    Captures mean-reversion at short horizons.
    """

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"reversal_{self.lookback}"

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        return -returns.rolling(self.lookback).sum()
