# Nepal Agriculture ML System

Production-oriented, modular machine learning architecture for:
- Crop recommendation (top-3 with probabilities)
- Multi-risk detection (pest, disease, water stress, soil mismatch)
- Yield prediction with quantile uncertainty range (tons/hectare)
- Explainable + confidence-aware recommendations
- Risk-prioritized intervention planning for farmer actionability

## Project structure

```text
.
├── data/
│   ├── sample_artifacts/
│   └── schemas/sample_dataset_schema.csv
├── scripts/
│   ├── train.py
│   └── infer_example.py
├── src/agri_ml/
│   ├── config/settings.py
│   ├── data/dataset.py
│   ├── evaluation/metrics.py
│   ├── features/engineering.py
│   ├── inference/predictor.py
│   ├── models/trainers.py
│   └── pipeline.py
└── pyproject.toml
```

## Quick start

```bash
python -m pip install -e .
python scripts/train.py
python scripts/infer_example.py
```

## Output JSON format

```json
{
  "recommended_crops": [
    {"name": "Rice", "confidence": 0.87},
    {"name": "Maize", "confidence": 0.72}
  ],
  "risks": {
    "pest": "medium",
    "disease": "low",
    "water_stress": "high"
  },
  "yield_prediction": "2.5–3.2 tons/hectare",
  "explanation": "...",
  "next_steps": ["..."],
  "confidence_score": 0.82
}
```

## Notes for Nepal deployment
- Plug real weather API features (DHM Nepal, NASA POWER, OpenWeather) into dataset columns.
- Map provinces and districts to historical crop/yield priors.
- Keep a farmer feedback loop for retraining and calibration.
- Use SHAP in dashboard/service layer for per-farm explanation if needed.
