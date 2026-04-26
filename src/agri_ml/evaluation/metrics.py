from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)


def evaluate_crop_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_risk_model(y_true: pd.DataFrame, y_pred: np.ndarray, columns: list[str]) -> Dict[str, object]:
    scores = {}
    for i, col in enumerate(columns):
        scores[col] = {
            "accuracy": float(accuracy_score(y_true[col], y_pred[:, i])),
            "precision_weighted": float(precision_score(y_true[col], y_pred[:, i], average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true[col], y_pred[:, i], average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true[col], y_pred[:, i], average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true[col], y_pred[:, i]).tolist(),
        }
    return scores


def evaluate_yield_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"rmse": float(rmse)}


def region_wise_crop_accuracy(df_with_preds: pd.DataFrame, region_col: str = "province") -> Dict[str, float]:
    grouped = df_with_preds.groupby(region_col).apply(
        lambda part: accuracy_score(part["recommended_crop"], part["crop_pred"])
    )
    return {k: float(v) for k, v in grouped.to_dict().items()}


def crop_specific_accuracy(df_with_preds: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for crop, part in df_with_preds.groupby("recommended_crop"):
        out[crop] = float((part["recommended_crop"] == part["crop_pred"]).mean())
    return out
