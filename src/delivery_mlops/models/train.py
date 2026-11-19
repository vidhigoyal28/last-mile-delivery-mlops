
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(model, X, y, dataset_name):
    """Evaluate a regression model using MAE, RMSE and R2."""

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = mean_squared_error(y, predictions) ** 0.5
    r2 = r2_score(y, predictions)

    print(f"\n{dataset_name} RESULTS")
    print("-" * 45)
    print(f"MAE  : {mae:.3f} minutes")
    print(f"RMSE : {rmse:.3f} minutes")
    print(f"R2   : {r2:.4f}")

    return {
        "dataset": dataset_name,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def train_baseline():

    # 1. Project directories
    project_root = Path(__file__).resolve().parents[3]

    split_dir = project_root / "data" / "processed" / "splits"
    model_dir = project_root / "models"
    artifact_dir = project_root / "artifacts"

    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("DELIVERY DURATION PREDICTION - BASELINE TRAINING")
    print("=" * 55)

    # 2. Load datasets
    print("\nLoading datasets...")

    train_df = pd.read_parquet(split_dir / "train.parquet")
    val_df = pd.read_parquet(split_dir / "validation.parquet")
    test_df = pd.read_parquet(split_dir / "test.parquet")

    print(f"Training records   : {len(train_df):,}")
    print(f"Validation records : {len(val_df):,}")
    print(f"Testing records    : {len(test_df):,}")

    # 3. Separate features and target
    target = "target_duration"

    # accept_datetime is used for chronological splitting,
    # so it must not be given to the model.
    excluded_columns = [target, "accept_datetime"]

    feature_columns = [
        col for col in train_df.columns
        if col not in excluded_columns
    ]

    X_train = train_df[feature_columns]
    y_train = train_df[target]

    X_val = val_df[feature_columns]
    y_val = val_df[target]

    X_test = test_df[feature_columns]
    y_test = test_df[target]

    print("\nFeatures used:")
    print(feature_columns)

    # 4. Train baseline model
    print("\nTraining Dummy Regressor...")

    model = DummyRegressor(strategy="median")
    model.fit(X_train, y_train)

    print("Baseline training completed!")

    # 5. Evaluate on validation data
    validation_metrics = evaluate_model(
        model,
        X_val,
        y_val,
        "VALIDATION"
    )

    # 6. Evaluate on test data
    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "TEST"
    )

    # 7. Save trained model
    model_path = model_dir / "baseline_model.joblib"
    joblib.dump(model, model_path)

    # 8. Save evaluation metrics
    metrics_path = artifact_dir / "baseline_metrics.json"

    metrics = {
        "model": "DummyRegressor",
        "strategy": "median",
        "features": feature_columns,
        "validation": validation_metrics,
        "test": test_metrics
    }

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "=" * 55)
    print("BASELINE TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print(f"Model saved   : {model_path}")
    print(f"Metrics saved : {metrics_path}")


if __name__ == "__main__":
    train_baseline()
