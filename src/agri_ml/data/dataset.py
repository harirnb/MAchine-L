from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agri_ml.config.settings import AppConfig


@dataclass
class DatasetBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_crop_train: pd.Series
    y_crop_val: pd.Series
    y_crop_test: pd.Series
    y_risk_train: pd.DataFrame
    y_risk_val: pd.DataFrame
    y_risk_test: pd.DataFrame
    y_yield_train: pd.Series
    y_yield_val: pd.Series
    y_yield_test: pd.Series


def create_synthetic_nepal_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    provinces = ["Koshi", "Madhesh", "Bagmati", "Gandaki", "Lumbini", "Karnali", "Sudurpashchim"]
    seasons = ["monsoon", "winter", "spring"]
    soil_textures = ["sandy", "loam", "clay", "silty"]
    crops = ["rice", "maize", "wheat", "millet", "lentil", "mustard", "potato"]

    df = pd.DataFrame(
        {
            "province": rng.choice(provinces, n_samples),
            "district": rng.choice(["Chitwan", "Kaski", "Kailali", "Morang", "Dang", "Jhapa"], n_samples),
            "gps_lat": rng.normal(28.2, 0.7, n_samples),
            "gps_lon": rng.normal(84.0, 1.5, n_samples),
            "soil_ph": np.clip(rng.normal(6.6, 0.7, n_samples), 4.8, 8.3),
            "soil_texture": rng.choice(soil_textures, n_samples),
            "soil_fertility_index": np.clip(rng.beta(4, 2, n_samples), 0, 1),
            "season": rng.choice(seasons, n_samples, p=[0.45, 0.3, 0.25]),
            "weather_temp_avg_30d": rng.normal(24, 5, n_samples),
            "weather_rainfall_30d_mm": np.clip(rng.normal(180, 120, n_samples), 0, None),
            "weather_humidity_avg_30d": np.clip(rng.normal(70, 12, n_samples), 30, 98),
            "weather_temp_avg_90d": rng.normal(22, 4.2, n_samples),
            "weather_rainfall_90d_mm": np.clip(rng.normal(450, 230, n_samples), 0, None),
            "weather_humidity_avg_90d": np.clip(rng.normal(68, 10, n_samples), 35, 98),
            "current_crop": rng.choice(crops, n_samples),
            "farm_size_ha": np.clip(rng.lognormal(mean=0.2, sigma=0.7, size=n_samples), 0.05, 20),
            "water_availability_index": np.clip(rng.beta(3, 2, n_samples), 0, 1),
            "symptom_text": rng.choice(
                [
                    "leaf-yellowing",
                    "spots on leaves",
                    "wilting in noon",
                    "stunted growth",
                    "healthy",
                ],
                n_samples,
            ),
            "symptom_pest_indicator": rng.integers(0, 2, n_samples),
            "symptom_disease_indicator": rng.integers(0, 2, n_samples),
        }
    )

    # Rule-guided pseudo labels for realistic signal.
    df["recommended_crop"] = np.where(
        (df["season"] == "monsoon") & (df["weather_rainfall_30d_mm"] > 180),
        "rice",
        np.where(df["season"] == "winter", "wheat", "maize"),
    )
    df.loc[df["soil_fertility_index"] < 0.3, "recommended_crop"] = "millet"
    df.loc[(df["soil_ph"] > 7.6) & (df["season"] == "winter"), "recommended_crop"] = "mustard"

    def risk_bucket(val: np.ndarray, low: float, high: float) -> pd.Series:
        return pd.cut(val, bins=[-np.inf, low, high, np.inf], labels=["low", "medium", "high"])  # type: ignore[return-value]

    df["risk_pest"] = risk_bucket(df["weather_humidity_avg_30d"] + df["symptom_pest_indicator"] * 15, 65, 80)
    df["risk_disease"] = risk_bucket(df["weather_humidity_avg_90d"] + df["symptom_disease_indicator"] * 18, 60, 78)
    df["risk_water_stress"] = risk_bucket(1 - df["water_availability_index"], 0.3, 0.6)
    df["risk_soil_mismatch"] = risk_bucket(np.abs(df["soil_ph"] - 6.6), 0.5, 1.1)

    noise = rng.normal(0, 0.22, n_samples)
    df["yield_ton_per_ha"] = (
        1.2
        + 1.8 * df["soil_fertility_index"]
        + 0.9 * df["water_availability_index"]
        + 0.002 * df["weather_rainfall_90d_mm"]
        - 0.35 * (df["risk_water_stress"] == "high").astype(int)
        + noise
    ).round(2)
    return df


def split_dataset(df: pd.DataFrame, config: AppConfig) -> DatasetBundle:
    y_crop = df[config.data.target_crop_col]
    y_risk = df[config.data.target_risk_cols]
    y_yield = df[config.data.target_yield_col]
    X = df.drop(columns=[config.data.target_crop_col, *config.data.target_risk_cols, config.data.target_yield_col])

    X_train, X_test, y_crop_train, y_crop_test, y_risk_train, y_risk_test, y_yield_train, y_yield_test = train_test_split(
        X,
        y_crop,
        y_risk,
        y_yield,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
        stratify=y_crop,
    )

    val_fraction_of_train = config.training.val_size / (1 - config.training.test_size)
    X_train, X_val, y_crop_train, y_crop_val, y_risk_train, y_risk_val, y_yield_train, y_yield_val = train_test_split(
        X_train,
        y_crop_train,
        y_risk_train,
        y_yield_train,
        test_size=val_fraction_of_train,
        random_state=config.training.random_state,
        stratify=y_crop_train,
    )

    return DatasetBundle(
        X_train, X_val, X_test,
        y_crop_train, y_crop_val, y_crop_test,
        y_risk_train, y_risk_val, y_risk_test,
        y_yield_train, y_yield_val, y_yield_test,
    )


def missing_summary(df: pd.DataFrame) -> Dict[str, float]:
    return (df.isna().mean() * 100).round(2).to_dict()
