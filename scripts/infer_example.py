from __future__ import annotations

import json

from agri_ml.pipeline import build_inference_service

sample_input = {
    "province": "Bagmati",
    "district": "Chitwan",
    "gps_lat": 27.529,
    "gps_lon": 84.354,
    "soil_ph": 6.4,
    "soil_texture": "loam",
    "soil_fertility_index": 0.71,
    "season": "monsoon",
    "weather_temp_avg_30d": 27.5,
    "weather_rainfall_30d_mm": 305.0,
    "weather_humidity_avg_30d": 84.0,
    "weather_temp_avg_90d": 25.1,
    "weather_rainfall_90d_mm": 680.0,
    "weather_humidity_avg_90d": 79.2,
    "current_crop": "rice",
    "farm_size_ha": 1.3,
    "water_availability_index": 0.76,
    "symptom_text": "mild leaf spots",
    "symptom_pest_indicator": 1,
    "symptom_disease_indicator": 1,
}

if __name__ == "__main__":
    service = build_inference_service()
    result = service.predict(sample_input)
    print(json.dumps(result, indent=2))
