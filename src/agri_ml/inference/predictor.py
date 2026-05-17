from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from agri_ml.config.settings import AppConfig
from agri_ml.features.engineering import add_engineered_features
from agri_ml.models.trainers import TrainedModels
from agri_ml.utils.explainability import rule_text_explanation, shap_like_local_explanation


@dataclass
class InferenceService:
    models: TrainedModels
    config: AppConfig
    crop_labels: List[str]

    def _risk_label(self, probs: np.ndarray) -> str:
        idx = int(np.argmax(probs))
        return self.config.inference.risk_classes[idx]

    def _confidence(self, top_prob: float, spread: float) -> float:
        score = 0.7 * top_prob + 0.3 * (1 - spread)
        return float(np.clip(score, 0, 1))

    def _risk_priority(self, risk_label: str) -> int:
        order = {"low": 1, "medium": 2, "high": 3}
        return order.get(risk_label.lower(), 0)

    def _interventions_from_risks(self, risks: Dict[str, str], confidence: float) -> List[str]:
        steps = [
            "Use certified seeds for the top recommended crop.",
            "Re-check soil pH and moisture before the next irrigation cycle.",
        ]

        if self._risk_priority(risks["pest"]) >= 2:
            steps.append("Set pheromone/light traps this week and scout leaves every 3 days.")
        if self._risk_priority(risks["disease"]) >= 2:
            steps.append("Apply a preventive disease management schedule and remove infected plants early.")
        if self._risk_priority(risks["water_stress"]) >= 2:
            steps.append("Shift to split irrigation and mulching to stabilize root-zone moisture.")
        if self._risk_priority(risks["soil_mismatch"]) >= 2:
            steps.append("Run a soil test and add targeted nutrient corrections before the next cycle.")

        if confidence < self.config.inference.low_conf_threshold:
            steps.append("Low confidence: validate recommendation with a local agronomy officer.")

        return steps

    def predict(self, input_payload: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([input_payload])
        df_feat = add_engineered_features(df)

        crop_probs = self.models.crop_model.predict_proba(df_feat)[0]
        top_idx = np.argsort(crop_probs)[::-1][: self.config.inference.top_k_crops]
        recs = [
            {"name": self.crop_labels[i].title(), "confidence": round(float(crop_probs[i]), 2)}
            for i in top_idx
        ]

        risk_proba_all = self.models.risk_model.predict_proba(df_feat)
        risk_labels = {
            "pest": self._risk_label(risk_proba_all[0][0]),
            "disease": self._risk_label(risk_proba_all[1][0]),
            "water_stress": self._risk_label(risk_proba_all[2][0]),
            "soil_mismatch": self._risk_label(risk_proba_all[3][0]),
        }

        y_pred = float(self.models.yield_model.predict(df_feat)[0])
        low_q = float(self.models.yield_lower_model.predict(df_feat)[0])
        high_q = float(self.models.yield_upper_model.predict(df_feat)[0])
        low = max(0.0, min(low_q, y_pred))
        high = max(y_pred, high_q)

        top_prob = float(crop_probs[top_idx[0]])
        spread = float(crop_probs[top_idx[0]] - crop_probs[top_idx[1]]) if len(top_idx) > 1 else 0.0
        confidence = self._confidence(top_prob, spread)

        transformed = self.models.crop_model.named_steps["prep"].transform(df_feat)
        feature_names = self.models.crop_model.named_steps["prep"].get_feature_names_out().tolist()
        local_features = shap_like_local_explanation(
            self.models.crop_model.named_steps["model"], transformed[0], feature_names
        )
        explanation = rule_text_explanation(local_features)

        next_steps = self._interventions_from_risks(risk_labels, confidence)

        return {
            "recommended_crops": recs,
            "risks": risk_labels,
            "yield_prediction": f"{low:.2f}–{high:.2f} tons/hectare",
            "explanation": explanation,
            "next_steps": next_steps,
            "confidence_score": round(confidence, 2),
        }
