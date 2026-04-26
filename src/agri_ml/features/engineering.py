from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Season encoding friendly numeric features.
    season_map = {"winter": 0, "spring": 1, "monsoon": 2}
    out["season_code"] = out["season"].map(season_map).fillna(1)

    # Soil quality score.
    out["soil_quality_score"] = (
        0.45 * out["soil_fertility_index"].fillna(out["soil_fertility_index"].median())
        + 0.35 * (1 - np.minimum(np.abs(out["soil_ph"].fillna(6.6) - 6.6) / 2, 1))
        + 0.20 * out["water_availability_index"].fillna(out["water_availability_index"].median())
    )

    # Weather aggregates (30d vs 90d behavior).
    out["rainfall_ratio_30_90"] = out["weather_rainfall_30d_mm"] / (out["weather_rainfall_90d_mm"] + 1)
    out["temp_shift_30_90"] = out["weather_temp_avg_30d"] - out["weather_temp_avg_90d"]
    out["humidity_shift_30_90"] = out["weather_humidity_avg_30d"] - out["weather_humidity_avg_90d"]

    # Crop compatibility proxy using known crop needs.
    crop_pref = {
        "rice": (200, 85),
        "maize": (120, 70),
        "wheat": (80, 60),
        "millet": (70, 55),
        "lentil": (65, 55),
        "mustard": (75, 58),
        "potato": (95, 68),
    }
    rain_pref = out["current_crop"].map(lambda c: crop_pref.get(c, (100, 65))[0]).fillna(100)
    hum_pref = out["current_crop"].map(lambda c: crop_pref.get(c, (100, 65))[1]).fillna(65)
    out["crop_compatibility_score"] = 1 - np.minimum(
        np.abs(out["weather_rainfall_30d_mm"] - rain_pref) / 300
        + np.abs(out["weather_humidity_avg_30d"] - hum_pref) / 100,
        1,
    )

    # Simple text-based signal extraction.
    out["symptom_severity"] = out["symptom_text"].fillna("").str.contains("spots|wilting|stunted", case=False).astype(int)
    return out


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = df.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )
