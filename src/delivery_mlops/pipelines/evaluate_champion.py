
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = ROOT / "models/tuned_xgboost_model.joblib"
HOLDOUT_FILE = ROOT / "data/incoming/incoming_holdout.parquet"

REPORT_FILE = (
    ROOT / "artifacts/quality_monitoring/champion_evaluation.json"
)

FEATURES = [
    "city", "region_id", "aoi_type",
    "accept_hour", "accept_dayofweek",
    "accept_month", "distance_km",
    "lat", "lng"
]

TARGET = "target_duration"


def evaluate_champion():

    print("Loading champion model...")
    model = joblib.load(MODEL_FILE)

    print("Loading incoming evaluation data...")
    df = pd.read_parquet(HOLDOUT_FILE)

    X = df[FEATURES]
    y_true = df[TARGET]

    print("Generating predictions...")
    predictions = model.predict(X)

    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)

    report = {
        "model_name": "champion_tuned_xgboost",
        "evaluation_records": int(len(df)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.write_text(
        json.dumps(report, indent=4)
    )

    print("\n========== CHAMPION RESULTS ==========")
    print(json.dumps(report, indent=4))

    print("\nChampion evaluation completed!")


if __name__ == "__main__":
    evaluate_champion()
