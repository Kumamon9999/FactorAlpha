import pandas as pd

from .base import BaseFactor


class PrecomputedFactor(BaseFactor):
    """
    Wrap an existing date x ticker score matrix so it can be used by
    BacktestEngine like any other factor.
    """

    def __init__(self, name: str, scores: pd.DataFrame):
        if scores.empty:
            raise ValueError("scores must be a non-empty DataFrame")
        self._name = name
        self.scores = scores.sort_index()

    @property
    def name(self) -> str:
        return self._name

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        return self.scores.reindex(index=returns.index, columns=returns.columns)
