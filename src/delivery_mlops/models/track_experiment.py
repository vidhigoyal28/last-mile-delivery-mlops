
import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "models" / "tuned_xgboost_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "tuned_xgboost_metrics.json"

# Use SQLite as the MLflow tracking backend
DB_PATH = ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")

# Create or select experiment
mlflow.set_experiment("Last_Mile_Delivery_Duration")

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

def log_metrics(data, prefix=""):
    if isinstance(data, dict):
        for key, value in data.items():
            log_metrics(value, f"{prefix}{key}_")
    elif isinstance(data, (int, float)):
        mlflow.log_metric(prefix.rstrip("_"), data)

with mlflow.start_run(run_name="Tuned_XGBoost"):
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.03)
    mlflow.log_param("min_child_weight", 3)
    mlflow.log_param("subsample", 1.0)
    mlflow.log_param("colsample_bytree", 0.8)

    log_metrics(metrics)

    mlflow.log_artifact(str(MODEL_PATH))
    mlflow.log_artifact(str(METRICS_PATH))

print("\nMLflow experiment tracking completed!")
print("Tracking database:", DB_PATH)
