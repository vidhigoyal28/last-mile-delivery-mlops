
from pathlib import Path
import json
import time
import joblib
import pandas as pd

from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[3]

CHAMPION_FILE = ROOT / "models" / "tuned_xgboost_model.joblib"
ORIGINAL_TRAIN_FILE = ROOT / "data" / "processed" / "splits" / "train.parquet"
INCOMING_TRAIN_FILE = ROOT / "data" / "incoming" / "incoming_train.parquet"

CANDIDATE_FILE = (
    ROOT / "models" / "candidates" / "candidate_xgboost_model.joblib"
)

REPORT_FILE = (
    ROOT / "artifacts" / "quality_monitoring" / "candidate_training.json"
)

FEATURES = [
    "city", "region_id", "aoi_type",
    "accept_hour", "accept_dayofweek",
    "accept_month", "distance_km",
    "lat", "lng"
]

TARGET = "target_duration"


def train_candidate():

    print("Loading original training data...")
    original = pd.read_parquet(ORIGINAL_TRAIN_FILE)

    print("Loading incoming training batch...")
    incoming = pd.read_parquet(INCOMING_TRAIN_FILE)

    # Combine original and incoming labelled data
    combined = pd.concat(
        [original, incoming],
        ignore_index=True
    )

    # Keep only model features and target
    combined = combined[FEATURES + [TARGET]]

    # Target must be available for supervised training
    combined = combined.dropna(subset=[TARGET])

    X_train = combined[FEATURES]
    y_train = combined[TARGET]

    print("\n========== TRAINING DATA ==========")
    print("Original records:", len(original))
    print("Incoming records:", len(incoming))
    print("Combined records:", len(combined))

    # Load and clone the existing champion pipeline.
    # This retains the same preprocessing and model configuration.
    print("\nLoading champion pipeline...")
    champion = joblib.load(CHAMPION_FILE)

    candidate = clone(champion)

    print("\nTraining candidate XGBoost...")
    start_time = time.time()

    candidate.fit(X_train, y_train)

    training_seconds = time.time() - start_time

    # Save candidate separately; do not overwrite champion
    CANDIDATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(candidate, CANDIDATE_FILE)

    report = {
        "model_name": "candidate_xgboost",
        "original_training_records": int(len(original)),
        "incoming_training_records": int(len(incoming)),
        "combined_training_records": int(len(combined)),
        "training_duration_seconds": round(training_seconds, 2),
        "candidate_model_path": str(CANDIDATE_FILE.relative_to(ROOT)),
        "status": "trained"
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.write_text(
        json.dumps(report, indent=4)
    )

    print("\n========== CANDIDATE TRAINING REPORT ==========")
    print(json.dumps(report, indent=4))

    print("\nCandidate model saved successfully!")

    # Optional MLflow tracking
    try:
        import mlflow

        mlflow.set_tracking_uri(
            f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
        )

        mlflow.set_experiment("Last_Mile_Delivery_Duration")

        with mlflow.start_run(
            run_name="assessment_candidate_training"
        ):
            mlflow.log_params({
                "training_records": len(combined),
                "incoming_records": len(incoming),
                "algorithm": "XGBoost",
                "model_type": "candidate"
            })

            mlflow.log_metric(
                "training_duration_seconds",
                training_seconds
            )

            mlflow.log_artifact(str(CANDIDATE_FILE))
            mlflow.log_artifact(str(REPORT_FILE))

        print("Candidate training tracked in MLflow.")

    except ImportError:
        print("MLflow not installed; skipping optional tracking.")

    except Exception as exc:
        print("Candidate saved, but MLflow tracking failed:", exc)


if __name__ == "__main__":
    train_candidate()
