import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..base import BaseFactor


def _build_features(returns: pd.DataFrame) -> dict:
    """Compute lagged return and volatility features for each stock."""
    return {
        "ret_1d":    returns,
        "ret_5d":    returns.rolling(5).sum(),
        "ret_21d":   returns.rolling(21).sum(),
        "ret_63d":   returns.rolling(63).sum(),
        "vol_20d":   returns.rolling(20).std() * np.sqrt(252),
        "vol_60d":   returns.rolling(60).std() * np.sqrt(252),
        "vol_ratio": (returns.rolling(20).std() / (returns.rolling(60).std() + 1e-9)),
    }


def _collect_panel(feat_dict, feat_names, fwd_ret, start, end):
    """Stack (date, stock) training rows from index range [start, end)."""
    X_rows, y_rows = [], []
    for t in range(start, end):
        x_mat = np.stack([feat_dict[fn].iloc[t].values for fn in feat_names], axis=1)
        y_vec = fwd_ret.iloc[t].values
        valid = ~np.isnan(x_mat).any(axis=1) & ~np.isnan(y_vec)
        if valid.sum() < 5:
            continue
        X_rows.append(x_mat[valid])
        y_rows.append(y_vec[valid])
    if not X_rows:
        return None, None
    return np.vstack(X_rows), np.concatenate(y_rows)


class RidgeAlphaFactor(BaseFactor):
    """
    Cross-sectional Ridge regression alpha factor.

    Features per (date, stock): lagged returns [1d/5d/21d/63d], realised vol
    [20d/60d], and vol-regime ratio. Target: `forward`-day ahead return.

    When the input window is long (> 2 × min_obs), predictions are computed
    in a rolling expanding-window fashion for every date — suitable for
    FactorResearch IC evaluation. When short (BacktestEngine window), only
    the last date is predicted.

    Parameters
    ----------
    forward : int
        Prediction horizon (days).
    alpha : float
        Ridge regularisation strength.
    min_obs : int
        Minimum training dates before predicting.
    """

    def __init__(self, forward: int = 1, alpha: float = 1.0, min_obs: int = 63):
        self.forward = forward
        self.alpha = alpha
        self.min_obs = min_obs

    @property
    def name(self) -> str:
        return f"ridge_alpha_{self.forward}_{self.alpha}"

    def _predict_at(self, feat_dict, feat_names, scaler, model, t):
        x = np.stack([feat_dict[fn].iloc[t].values for fn in feat_names], axis=1)
        valid = ~np.isnan(x).any(axis=1)
        if valid.sum() < 2:
            return None, None
        return model.predict(scaler.transform(x[valid])), valid

    def compute(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        scores = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        feat_dict = _build_features(returns)
        feat_names = list(feat_dict.keys())
        T = len(returns)
        rolling_mode = T > 2 * self.min_obs

        if T < self.min_obs + self.forward + 1:
            return scores

        fwd_ret = returns.shift(-self.forward)

        if rolling_mode:
            # expanding-window: for each date t, train on [0..t-1], predict at t
            for t in range(self.min_obs, T):
                X_tr, y_tr = _collect_panel(feat_dict, feat_names, fwd_ret, 0, t)
                if X_tr is None:
                    continue
                scaler = StandardScaler()
                model = Ridge(alpha=self.alpha)
                model.fit(scaler.fit_transform(X_tr), y_tr)
                preds, valid = self._predict_at(feat_dict, feat_names, scaler, model, t)
                if preds is None:
                    continue
                s = pd.Series(np.nan, index=returns.columns)
                s[returns.columns[valid]] = preds
                scores.iloc[t] = s
        else:
            # single prediction at last date (BacktestEngine use case)
            X_tr, y_tr = _collect_panel(feat_dict, feat_names, fwd_ret,
                                        self.min_obs, T - self.forward)
            if X_tr is None:
                return scores
            scaler = StandardScaler()
            model = Ridge(alpha=self.alpha)
            model.fit(scaler.fit_transform(X_tr), y_tr)
            preds, valid = self._predict_at(feat_dict, feat_names, scaler, model, T - 1)
            if preds is not None:
                s = pd.Series(np.nan, index=returns.columns)
                s[returns.columns[valid]] = preds
                scores.iloc[-1] = s

        return scores
