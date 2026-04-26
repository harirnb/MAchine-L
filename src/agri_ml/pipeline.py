from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import joblib
import pandas as pd

from agri_ml.config.settings import AppConfig
from agri_ml.data.dataset import create_synthetic_nepal_dataset, split_dataset
from agri_ml.evaluation.metrics import (
    crop_specific_accuracy,
    evaluate_crop_model,
    evaluate_risk_model,
    evaluate_yield_model,
    region_wise_crop_accuracy,
)
from agri_ml.features.engineering import add_engineered_features
from agri_ml.inference.predictor import InferenceService
from agri_ml.models.trainers import train_all_models


def run_training_pipeline(config: AppConfig | None = None, data: pd.DataFrame | None = None) -> Dict[str, Any]:
    config = config or AppConfig()
    raw_df = data.copy() if data is not None else create_synthetic_nepal_dataset(random_state=config.training.random_state)
    feat_df = add_engineered_features(raw_df)
    bundle = split_dataset(feat_df, config)

    trained = train_all_models(
        bundle.X_train,
        bundle.y_crop_train,
        bundle.y_risk_train,
        bundle.y_yield_train,
        config,
    )

    crop_pred = trained.crop_model.predict(bundle.X_test)
    risk_pred = trained.risk_model.predict(bundle.X_test)
    yield_pred = trained.yield_model.predict(bundle.X_test)

    eval_crop = evaluate_crop_model(bundle.y_crop_test, crop_pred)
    eval_risk = evaluate_risk_model(bundle.y_risk_test, risk_pred, config.data.target_risk_cols)
    eval_yield = evaluate_yield_model(bundle.y_yield_test, yield_pred)

    analysis_df = bundle.X_test.copy()
    analysis_df["recommended_crop"] = bundle.y_crop_test.values
    analysis_df["crop_pred"] = crop_pred

    outputs = {
        "config": asdict(config),
        "metrics": {
            "crop": eval_crop,
            "risk": eval_risk,
            "yield": eval_yield,
            "region_wise_crop_accuracy": region_wise_crop_accuracy(analysis_df),
            "crop_specific_accuracy": crop_specific_accuracy(analysis_df),
        },
        "models": trained,
    }
    return outputs


def save_artifacts(outputs: Dict[str, Any], model_path: str = "data/sample_artifacts/model_bundle.joblib") -> None:
    model_bundle = {
        "config": outputs["config"],
        "models": outputs["models"],
    }
    joblib.dump(model_bundle, model_path)


def build_inference_service(model_path: str = "data/sample_artifacts/model_bundle.joblib") -> InferenceService:
    bundle = joblib.load(model_path)
    config = AppConfig()
    config.inference = config.inference

    crop_labels = list(bundle["models"].crop_model.named_steps["model"].classes_)
    return InferenceService(models=bundle["models"], config=config, crop_labels=crop_labels)
