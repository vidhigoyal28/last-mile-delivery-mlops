
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = (
    ROOT / "models/candidates/candidate_xgboost_model.joblib"
)

HOLDOUT_FILE = (
    ROOT / "data/incoming/incoming_holdout.parquet"
)

REPORT_FILE = (
    ROOT / "artifacts/quality_monitoring/candidate_evaluation.json"
)

FEATURES = [
    "city", "region_id", "aoi_type",
    "accept_hour", "accept_dayofweek",
    "accept_month", "distance_km",
    "lat", "lng"
]

TARGET = "target_duration"


def evaluate_candidate():

    print("Loading candidate model...")
    model = joblib.load(MODEL_FILE)

    print("Loading evaluation holdout...")
    df = pd.read_parquet(HOLDOUT_FILE)

    X = df[FEATURES]
    y_true = df[TARGET]

    print("Generating candidate predictions...")
    predictions = model.predict(X)

    report = {
        "model_name": "candidate_xgboost",
        "evaluation_records": int(len(df)),
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, predictions))
        ),
        "R2": float(r2_score(y_true, predictions))
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.write_text(
        json.dumps(report, indent=4)
    )

    print("\n========== CANDIDATE RESULTS ==========")
    print(json.dumps(report, indent=4))

    print("\nCandidate evaluation completed!")


if __name__ == "__main__":
    evaluate_candidate()
