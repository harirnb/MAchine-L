from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from agri_ml.config.settings import AppConfig
from agri_ml.features.engineering import build_preprocessor


@dataclass
class TrainedModels:
    crop_model: Pipeline
    risk_model: Pipeline
    yield_model: Pipeline
    yield_lower_model: Pipeline
    yield_upper_model: Pipeline


def _crop_estimator(random_state: int) -> LGBMClassifier:
    return LGBMClassifier(objective="multiclass", random_state=random_state, n_estimators=250)


def _risk_estimator(random_state: int) -> MultiOutputClassifier:
    base = XGBClassifier(
        objective="multi:softprob",
        random_state=random_state,
        eval_metric="mlogloss",
        max_depth=5,
        n_estimators=220,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    return MultiOutputClassifier(base)


def _yield_estimator(random_state: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
    )


def train_all_models(X_train: pd.DataFrame, y_crop: pd.Series, y_risk: pd.DataFrame, y_yield: pd.Series, config: AppConfig) -> TrainedModels:
    preprocessor = build_preprocessor(X_train)

    crop_pipe = Pipeline([("prep", preprocessor), ("model", _crop_estimator(config.training.random_state))])
    risk_pipe = Pipeline([("prep", preprocessor), ("model", _risk_estimator(config.training.random_state))])
    yield_pipe = Pipeline([("prep", preprocessor), ("model", _yield_estimator(config.training.random_state))])
    yield_lower_pipe = Pipeline(
        [("prep", preprocessor), ("model", GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=config.training.random_state))]
    )
    yield_upper_pipe = Pipeline(
        [("prep", preprocessor), ("model", GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=config.training.random_state))]
    )

    crop_param_grid = {
        "model__num_leaves": [15, 31, 63],
        "model__max_depth": [-1, 8, 12],
        "model__learning_rate": [0.03, 0.05, 0.08],
    }
    crop_search = RandomizedSearchCV(
        crop_pipe,
        crop_param_grid,
        n_iter=config.training.n_iter_search,
        cv=config.training.cv_folds,
        scoring="f1_weighted",
        random_state=config.training.random_state,
        n_jobs=-1,
    )
    crop_search.fit(X_train, y_crop)

    risk_pipe.fit(X_train, y_risk)
    yield_pipe.fit(X_train, y_yield)
    yield_lower_pipe.fit(X_train, y_yield)
    yield_upper_pipe.fit(X_train, y_yield)

    return TrainedModels(
        crop_model=crop_search.best_estimator_,
        risk_model=risk_pipe,
        yield_model=yield_pipe,
        yield_lower_model=yield_lower_pipe,
        yield_upper_model=yield_upper_pipe,
    )


def model_feature_importance(crop_model: Pipeline, feature_names: np.ndarray) -> Dict[str, float]:
    lgbm = crop_model.named_steps["model"]
    importances = lgbm.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return {k: float(v) for k, v in pairs[:10]}
