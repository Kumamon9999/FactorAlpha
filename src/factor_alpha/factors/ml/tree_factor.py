import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from ..base import BaseFactor
from .ridge_factor import _build_features, _collect_panel


def _predict_at_tree(feat_dict, feat_names, model, t):
    x = np.stack([feat_dict[fn].iloc[t].values for fn in feat_names], axis=1)
    valid = ~np.isnan(x).any(axis=1)
    if valid.sum() < 2:
        return None, None
    return model.predict(x[valid]), valid


class GradientBoostFactor(BaseFactor):
    """
    Gradient Boosted Trees cross-sectional alpha factor.

    Same lagged-return + volatility features as RidgeAlphaFactor, but fits a
    GradientBoostingRegressor that captures non-linear interactions (e.g. momentum
    behaves differently in low-vol vs high-vol regimes).

    Supports rolling expanding-window mode for FactorResearch evaluation and
    single-date mode for BacktestEngine.

    Parameters
    ----------
    forward : int
        Prediction horizon (days).
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Tree depth (keep shallow to limit overfitting).
    min_obs : int
        Minimum training dates before first prediction.
    """

    def __init__(
        self,
        forward: int = 1,
        n_estimators: int = 100,
        max_depth: int = 3,
        min_obs: int = 63,
    ):
        self.forward = forward
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_obs = min_obs

    @property
    def name(self) -> str:
        return f"gbm_alpha_{self.forward}_{self.n_estimators}"

    def _make_model(self):
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

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
            for t in range(self.min_obs, T):
                X_tr, y_tr = _collect_panel(feat_dict, feat_names, fwd_ret, 0, t)
                if X_tr is None:
                    continue
                model = self._make_model()
                model.fit(X_tr, y_tr)
                preds, valid = _predict_at_tree(feat_dict, feat_names, model, t)
                if preds is None:
                    continue
                s = pd.Series(np.nan, index=returns.columns)
                s[returns.columns[valid]] = preds
                scores.iloc[t] = s
        else:
            X_tr, y_tr = _collect_panel(feat_dict, feat_names, fwd_ret,
                                        self.min_obs, T - self.forward)
            if X_tr is None:
                return scores
            model = self._make_model()
            model.fit(X_tr, y_tr)
            preds, valid = _predict_at_tree(feat_dict, feat_names, model, T - 1)
            if preds is not None:
                s = pd.Series(np.nan, index=returns.columns)
                s[returns.columns[valid]] = preds
                scores.iloc[-1] = s

        return scores


class RandomForestFactor(BaseFactor):
    """
    Random Forest cross-sectional alpha factor.

    Ensemble of decision trees with bootstrap sampling. More robust to outliers
    than GradientBoostFactor; exposes `feature_importances_` after fitting.

    Supports rolling expanding-window mode for FactorResearch and single-date
    mode for BacktestEngine.

    Parameters
    ----------
    forward : int
        Prediction horizon (days).
    n_estimators : int
        Number of trees.
    max_depth : int
        Maximum tree depth.
    min_obs : int
        Minimum training dates before first prediction.
    """

    def __init__(
        self,
        forward: int = 1,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_obs: int = 63,
    ):
        self.forward = forward
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_obs = min_obs

    @property
    def name(self) -> str:
        return f"rf_alpha_{self.forward}_{self.n_estimators}"

    def _make_model(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=42,
        )

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
            last_model = None
            for t in range(self.min_obs, T):
                X_tr, y_tr = _collect_panel(feat_dict, feat_names, fwd_ret, 0, t)
                if X_tr is None:
                    continue
                model = self._make_model()
                model.fit(X_tr, y_tr)
                last_model = model
                preds, valid = _predict_at_tree(feat_dict, feat_names, model, t)
                if preds is None:
                    continue
                s = pd.Series(np.nan, index=returns.columns)
                s[returns.columns[valid]] = preds
                scores.iloc[t] = s
            if last_model is not None:
                self.feature_importances_ = dict(zip(feat_names, last_model.feature_importances_))
        else:
            X_tr, y_tr = _collect_panel(feat_dict, feat_names, fwd_ret,
                                        self.min_obs, T - self.forward)
            if X_tr is None:
                return scores
            model = self._make_model()
            model.fit(X_tr, y_tr)
            self.feature_importances_ = dict(zip(feat_names, model.feature_importances_))
            preds, valid = _predict_at_tree(feat_dict, feat_names, model, T - 1)
            if preds is not None:
                s = pd.Series(np.nan, index=returns.columns)
                s[returns.columns[valid]] = preds
                scores.iloc[-1] = s

        return scores
