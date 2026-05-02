from abc import ABC, abstractmethod

import pandas as pd


class BaseFactor(ABC):
    """
    A factor produces a (dates × tickers) DataFrame of raw scores.
    Higher score = stronger positive signal (long bias).
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            prices:  adjusted close prices, DatetimeIndex rows, ticker columns
            returns: log returns aligned to prices

        Returns:
            DataFrame of raw factor scores, same shape as returns (NaN where
            insufficient history exists).
        """
        ...
