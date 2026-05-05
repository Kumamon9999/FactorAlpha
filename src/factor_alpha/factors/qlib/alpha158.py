"""
Qlib Alpha158-inspired factor set computed from close prices and returns.

Alpha158 is Qlib's standard 158-feature benchmark. This module implements
the close-price-computable subset (~45 features across 5 windows) that fits
directly into FactorAlpha's BaseFactor interface without requiring qlib.init()
or Qlib's binary data format.

Full Alpha158 requires OHLCV; the features here use close + volume only, which
are the two fields universally available from all FactorAlpha data loaders.

Reference: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..base import BaseFactor
from ..ml.ridge_factor import _collect_panel

# Qlib's standard rolling windows (days)
_WINDOWS = [5, 10, 20, 30, 60]


def build_alpha158_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute a close-price (+volume) subset of Qlib's Alpha158 feature set.

    Each entry in the returned dict is a (dates × tickers) DataFrame
    on the same index/columns as `returns`.

    Features
    --------
    Per lagged horizon d ∈ {1, 5, 10, 20, 60}:
      - RET_<d>D : d-day cumulative log return (momentum)

    Per rolling window w ∈ {5, 10, 20, 30, 60}:
      - ROC_<w>   : rate-of-change = close/close[w days ago] − 1
      - MA_<w>    : rolling mean / close  (mean-reversion baseline)
      - STD_<w>   : rolling return std    (volatility regime)
      - MAX_<w>   : rolling high / close  (proximity to recent high)
      - MIN_<w>   : rolling low / close   (proximity to recent low)
      - RSV_<w>   : stochastic oscillator — where close sits in w-day range
      - QTLU_<w>  : 80th-percentile price / close
      - QTLD_<w>  : 20th-percentile price / close

    If `volumes` is provided, adds per-window:
      - VROC_<w>  : volume rate-of-change
      - VSTD_<w>  : rolling volume std / rolling volume mean

    Returns
    -------
    dict mapping feature name → (dates × tickers) DataFrame
    """
    close = prices
    feat: Dict[str, pd.DataFrame] = {}

    # Lagged return momentum
    for d in [1, 5, 10, 20, 60]:
        feat[f"RET_{d}D"] = returns.rolling(d).sum()

    for w in _WINDOWS:
        rolling_max = close.rolling(w).max()
        rolling_min = close.rolling(w).min()
        price_range = (rolling_max - rolling_min).replace(0, np.nan)

        feat[f"ROC_{w}"]  = close / close.shift(w) - 1
        feat[f"MA_{w}"]   = close.rolling(w).mean() / close.replace(0, np.nan)
        feat[f"STD_{w}"]  = returns.rolling(w).std()
        feat[f"MAX_{w}"]  = rolling_max / close.replace(0, np.nan)
        feat[f"MIN_{w}"]  = rolling_min / close.replace(0, np.nan)
        feat[f"RSV_{w}"]  = (close - rolling_min) / price_range
        feat[f"QTLU_{w}"] = close.rolling(w).quantile(0.8) / close.replace(0, np.nan)
        feat[f"QTLD_{w}"] = close.rolling(w).quantile(0.2) / close.replace(0, np.nan)

    if volumes is not None:
        vol = volumes.replace(0, np.nan)
        for w in _WINDOWS:
            feat[f"VROC_{w}"] = vol / vol.shift(w) - 1
            vol_mean = vol.rolling(w).mean().replace(0, np.nan)
            feat[f"VSTD_{w}"] = vol.rolling(w).std() / vol_mean

    return feat


class Alpha158Factor(BaseFactor):
    """
    Qlib Alpha158-inspired alpha factor using Ridge regression.

    Replaces RidgeAlphaFactor's 7-feature set with the ~45-feature Alpha158
    subset (multi-window RSV, quantiles, max/min ratios, momentum, vol).
    Produces cross-sectional return predictions via a single-shot Ridge fit
    on the full available history.

    Parameters
    ----------
    forward : int
        Prediction horizon in trading days.
    alpha : float
        Ridge regularization strength.
    min_obs : int
        Minimum training dates before making predictions.
    use_volume : bool
        Include volume-based features (VROC, VSTD). Requires a `volumes`
        DataFrame passed to compute(). Ignored if not available.
    """

    def __init__(
        self,
        forward: int = 1,
        alpha: float = 1.0,
        min_obs: int = 63,
        use_volume: bool = False,
    ):
        self.forward = forward
        self.alpha = alpha
        self.min_obs = min_obs
        self.use_volume = use_volume

    @property
    def name(self) -> str:
        return f"alpha158_{self.forward}_{self.alpha}"

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute cross-sectional Alpha158 scores.

        Parameters
        ----------
        prices  : adjusted close prices (T × N)
        returns : log returns aligned to prices (T × N)
        volumes : optional volume DataFrame for volume features (T × N)
        """
        scores = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        T = len(returns)

        if T < self.min_obs + self.forward + 1:
            return scores

        vol = volumes if self.use_volume and volumes is not None else None
        feat_dict = build_alpha158_features(prices, returns, vol)
        feat_names = list(feat_dict.keys())
        fwd_ret = returns.shift(-self.forward)

        X_tr, y_tr = _collect_panel(
            feat_dict, feat_names, fwd_ret, self.min_obs, T - self.forward
        )
        if X_tr is None:
            return scores

        scaler = StandardScaler()
        model = Ridge(alpha=self.alpha)
        model.fit(scaler.fit_transform(X_tr), y_tr)

        # Predict at last date
        x = np.stack([feat_dict[fn].iloc[-1].values for fn in feat_names], axis=1)
        valid = ~np.isnan(x).any(axis=1)
        if valid.sum() >= 2:
            preds = model.predict(scaler.transform(x[valid]))
            s = pd.Series(np.nan, index=returns.columns)
            s[returns.columns[valid]] = preds
            scores.iloc[-1] = s

        return scores


class QlibLGBFactor(BaseFactor):
    """
    LightGBM alpha factor using Qlib's Alpha158 feature set.

    Mirrors Qlib's LGBModel interface but fits inside FactorAlpha's
    walk-forward pipeline: trains on (features, forward_return) pairs
    over the lookback window and scores stocks at the last date.

    Parameters
    ----------
    forward : int
        Prediction horizon in trading days.
    num_leaves : int
        LightGBM num_leaves parameter.
    n_estimators : int
        Number of boosting rounds.
    min_obs : int
        Minimum training dates before making predictions.
    use_volume : bool
        Include volume-based features. Requires volumes in compute().
    """

    def __init__(
        self,
        forward: int = 1,
        num_leaves: int = 31,
        n_estimators: int = 100,
        min_obs: int = 63,
        use_volume: bool = False,
    ):
        try:
            import lightgbm  # noqa: F401
        except ImportError as e:
            raise ImportError("lightgbm is required for QlibLGBFactor. pip install lightgbm") from e

        self.forward = forward
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.min_obs = min_obs
        self.use_volume = use_volume

    @property
    def name(self) -> str:
        return f"qlib_lgb_{self.forward}_{self.n_estimators}"

    def _make_model(self):
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        scores = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        T = len(returns)

        if T < self.min_obs + self.forward + 1:
            return scores

        vol = volumes if self.use_volume and volumes is not None else None
        feat_dict = build_alpha158_features(prices, returns, vol)
        feat_names = list(feat_dict.keys())
        fwd_ret = returns.shift(-self.forward)

        X_tr, y_tr = _collect_panel(
            feat_dict, feat_names, fwd_ret, self.min_obs, T - self.forward
        )
        if X_tr is None:
            return scores

        model = self._make_model()
        # Pass feature names so LightGBM can match train/predict columns without warnings.
        X_tr_df = pd.DataFrame(X_tr, columns=feat_names)
        model.fit(X_tr_df, y_tr)
        self.feature_importances_ = dict(zip(feat_names, model.feature_importances_))

        # Predict at last date — pass as DataFrame to keep feature names consistent.
        x = np.stack([feat_dict[fn].iloc[-1].values for fn in feat_names], axis=1)
        valid = ~np.isnan(x).any(axis=1)
        if valid.sum() >= 2:
            x_df = pd.DataFrame(x[valid], columns=feat_names)
            preds = model.predict(x_df)
            s = pd.Series(np.nan, index=returns.columns)
            s[returns.columns[valid]] = preds
            scores.iloc[-1] = s

        return scores
