"""Two-stage freight rate model.

Stage 1  Ridge regression on log(rate) with a small set of interpretable,
         extrapolating drivers: distance elasticity, equipment premium,
         weight, market index, quarter-end ramp and a linear time trend.
Stage 2  LightGBM on the stage-1 residual, capturing lane geography and
         non-linear interactions, without features that cannot extrapolate.

Prediction = exp(stage1 + stage2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor, Ridge

from .features import GBM_FEATURES, LINEAR_FEATURES

GBM_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=42,
)
GBM_ROUNDS = 800
OUTLIER_LOG_THRESHOLD = np.log(1.6)  # flag labels >60% above / below expectation


def fit_outlier_detector(features: pd.DataFrame, log_rate: pd.Series) -> HuberRegressor:
    """Robust (Huber) linear fit of log(rate); used only to detect corrupt labels."""
    return HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500).fit(features[LINEAR_FEATURES], log_rate)


def apply_outlier_detector(detector: HuberRegressor, features: pd.DataFrame, log_rate: pd.Series) -> pd.Series:
    """Flag labels more than 60% above / below the detector's expectation.

    EDA shows two tight clusters at roughly 3-5x and 1/3-1/5x the expected rate
    (~1.4% of rows) and nothing in between, so the threshold is not sensitive.
    """
    residual = log_rate - detector.predict(features[LINEAR_FEATURES])
    return residual.abs() > OUTLIER_LOG_THRESHOLD


def flag_label_outliers(features: pd.DataFrame, log_rate: pd.Series) -> pd.Series:
    """Fit and apply on the same rows. Used for the final model (Jan-Oct) only."""
    return apply_outlier_detector(fit_outlier_detector(features, log_rate), features, log_rate)


XGB_PARAMS = dict(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, min_child_weight=10, tree_method="hist", random_state=42, n_jobs=2)
ET_PARAMS = dict(n_estimators=200, min_samples_leaf=3, max_features=0.7, n_jobs=2, random_state=42)
FINAL_RESIDUAL_MODELS = ("lightgbm", "xgboost", "extra_trees")


class _ResidualLearner:
    """One stage-2 learner on the Ridge residual. Extra Trees cannot take NaN, so
    missing weight is filled with the TRAINING median (and the missing flag is a feature)."""

    def __init__(self, kind: str, gbm_params: dict, gbm_rounds: int):
        self.kind, self.gbm_params, self.gbm_rounds = kind, gbm_params, gbm_rounds

    def fit(self, X: pd.DataFrame, r: pd.Series):
        if self.kind == "lightgbm":
            self.model_ = lgb.train(self.gbm_params, lgb.Dataset(X, r), self.gbm_rounds)
        elif self.kind == "xgboost":
            self.model_ = xgb.XGBRegressor(**XGB_PARAMS).fit(X, r)
        elif self.kind == "extra_trees":
            self.medians_ = X.median()
            self.model_ = ExtraTreesRegressor(**ET_PARAMS).fit(X.fillna(self.medians_), r)
        else:
            raise ValueError(self.kind)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "extra_trees":
            X = X.fillna(self.medians_)
        return self.model_.predict(X)


@dataclass
class TwoStageRateModel:
    """Stage 1: Ridge on log(rate). Stage 2: equal-weight average of residual learners.

    The submitted model uses residual_models = ("lightgbm", "xgboost", "extra_trees"),
    chosen in src/compare.py; ("lightgbm",) reproduces the original single-learner model.
    """
    ridge_alpha: float = 1e-3
    gbm_params: dict = field(default_factory=lambda: dict(GBM_PARAMS))
    gbm_rounds: int = GBM_ROUNDS
    use_gbm: bool = True
    residual_models: tuple = FINAL_RESIDUAL_MODELS

    def _residual_target(self, features, log_rate):
        return log_rate - self.linear_.predict(features[LINEAR_FEATURES])

    def fit(self, features: pd.DataFrame, log_rate: pd.Series) -> "TwoStageRateModel":
        self.linear_ = Ridge(alpha=self.ridge_alpha).fit(features[LINEAR_FEATURES], log_rate)
        if self.use_gbm:
            residual = self._residual_target(features, log_rate)
            self.learners_ = [_ResidualLearner(k, self.gbm_params, self.gbm_rounds).fit(features[GBM_FEATURES], residual)
                              for k in self.residual_models]
        return self

    @property
    def gbm_(self):
        """The LightGBM member (used for SHAP explanations)."""
        return next(l.model_ for l in self.learners_ if l.kind == "lightgbm")

    def predict_log(self, features: pd.DataFrame) -> np.ndarray:
        out = self.linear_.predict(features[LINEAR_FEATURES])
        if self.use_gbm:
            out = out + np.mean([l.predict(features[GBM_FEATURES]) for l in self.learners_], axis=0)
        return out

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(features))

    def linear_coefficients(self) -> dict[str, float]:
        coefs = dict(zip(LINEAR_FEATURES, self.linear_.coef_))
        coefs["intercept"] = float(self.linear_.intercept_)
        return coefs


class GBMOnlyModel:
    """Baseline: LightGBM directly on log(rate) (no trend handling)."""

    def __init__(self, rounds: int = 1500):
        self.rounds = rounds

    def fit(self, features, log_rate):
        params = dict(GBM_PARAMS, num_leaves=63)
        self.gbm_ = lgb.train(params, lgb.Dataset(features[GBM_FEATURES], log_rate), self.rounds)
        return self

    def predict(self, features):
        return np.exp(self.gbm_.predict(features[GBM_FEATURES]))
