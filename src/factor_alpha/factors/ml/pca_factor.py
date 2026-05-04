import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..base import BaseFactor


class PCAResidualFactor(BaseFactor):
    """
    PCA-based idiosyncratic residual factor.

    Fits PCA on the return matrix to extract K systematic (market/sector) components.
    Score = rolling mean of idiosyncratic residuals (actual - PCA reconstruction).

    Intuition: a stock persistently outperforming what its factor exposures predict
    has positive alpha; one persistently underperforming has negative alpha.

    Parameters
    ----------
    n_components : int
        Number of PCA components to remove (typically 3-10).
    signal_window : int
        Rolling window over which to average residuals into a score.
    """

    def __init__(self, n_components: int = 5, signal_window: int = 21):
        self.n_components = n_components
        self.signal_window = signal_window

    @property
    def name(self) -> str:
        return f"pca_residual_{self.n_components}_{self.signal_window}"

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        clean = returns.dropna(axis=1)
        n_comp = min(self.n_components, clean.shape[1] - 1, clean.shape[0] - 1)
        if n_comp < 1:
            return pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

        mu = clean.mean()
        sigma = clean.std().replace(0, 1)
        ret_std = (clean - mu) / sigma

        pca = PCA(n_components=n_comp)
        loadings = pca.fit_transform(ret_std.values)       # T × K
        reconstruction = pca.inverse_transform(loadings)   # T × N
        residuals = ret_std.values - reconstruction        # T × N

        resid_df = pd.DataFrame(residuals, index=clean.index, columns=clean.columns)
        scores = resid_df.rolling(
            self.signal_window, min_periods=max(1, self.signal_window // 2)
        ).mean()

        return scores.reindex(columns=returns.columns)
