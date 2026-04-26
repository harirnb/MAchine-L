from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DataConfig:
    target_crop_col: str = "recommended_crop"
    target_yield_col: str = "yield_ton_per_ha"
    target_risk_cols: List[str] = field(
        default_factory=lambda: [
            "risk_pest",
            "risk_disease",
            "risk_water_stress",
            "risk_soil_mismatch",
        ]
    )


@dataclass
class TrainingConfig:
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    cv_folds: int = 5
    n_iter_search: int = 20


@dataclass
class InferenceConfig:
    low_conf_threshold: float = 0.5
    risk_classes: List[str] = field(default_factory=lambda: ["low", "medium", "high"])
    top_k_crops: int = 3


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    province_to_region: Dict[str, str] = field(
        default_factory=lambda: {
            "Koshi": "Eastern",
            "Madhesh": "Central",
            "Bagmati": "Central",
            "Gandaki": "Western",
            "Lumbini": "Western",
            "Karnali": "Mid-West",
            "Sudurpashchim": "Far-West",
        }
    )
