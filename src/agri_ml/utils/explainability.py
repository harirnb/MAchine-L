from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def global_feature_importance_from_model(model, feature_names: list[str]) -> Dict[str, float]:
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        pairs = sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)
        return {k: float(v) for k, v in pairs[:10]}
    return {}


def shap_like_local_explanation(model, transformed_row: np.ndarray, feature_names: list[str]) -> Dict[str, float]:
    """Approximate local explanation when SHAP is not served in real-time.

    For production UI, replace with exact SHAP explainer object persisted at training.
    """
    if hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"):
        vals = model.feature_importances_ * np.abs(transformed_row.reshape(-1))
        idx = np.argsort(vals)[::-1][:5]
        return {feature_names[i]: float(vals[i]) for i in idx}
    return {}


def rule_text_explanation(top_features: Dict[str, float]) -> str:
    if not top_features:
        return "Recommendation based on seasonal, soil, and weather profile alignment."
    human = ", ".join(list(top_features.keys())[:3])
    return f"This crop is recommended because {human} match optimal conditions."
