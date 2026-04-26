from __future__ import annotations

import json
from pathlib import Path

from agri_ml.pipeline import run_training_pipeline, save_artifacts


if __name__ == "__main__":
    outputs = run_training_pipeline()
    save_artifacts(outputs)

    metrics_path = Path("data/sample_artifacts/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(outputs["metrics"], indent=2))

    print("Training complete.")
    print(json.dumps(outputs["metrics"], indent=2))
