
from pathlib import Path
import json

import joblib
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(model, X, y, name):
    predictions = model.predict(X)

    metrics = {
        "MAE": float(mean_absolute_error(y, predictions)),
        "RMSE": float(mean_squared_error(y, predictions) ** 0.5),
        "R2": float(r2_score(y, predictions))
    }

    print(f"\n{name} RESULTS")
    print("-" * 45)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


def train_tuned_xgboost():

    root = Path(__file__).resolve().parents[3]

    split_dir = root / "data" / "processed" / "splits"
    model_dir = root / "models"
    artifact_dir = root / "artifacts"

    model_dir.mkdir(exist_ok=True)
    artifact_dir.mkdir(exist_ok=True)

    print("Loading complete datasets...")

    train = pd.read_parquet(split_dir / "train.parquet")
    val = pd.read_parquet(split_dir / "validation.parquet")
    test = pd.read_parquet(split_dir / "test.parquet")

    target = "target_duration"
    excluded = [target, "accept_datetime"]

    features = [
        col for col in train.columns
        if col not in excluded
    ]

    categorical = ["city", "region_id", "aoi_type"]
    numerical = [
        col for col in features
        if col not in categorical
    ]

    X_train = train[features]
    y_train = train[target]

    X_val = val[features]
    y_val = val[target]

    X_test = test[features]
    y_test = test[target]

    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Test records:", len(X_test))

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical
            ),
            (
                "numerical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numerical
            )
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            min_child_weight=3,
            subsample=1.0,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=4,
            random_state=42
        ))
    ])

    print("\nTraining tuned XGBoost on full training data...")

    model.fit(X_train, y_train)

    print("Training completed!")

    # Validation is used to assess the selected configuration.
    validation_metrics = evaluate_model(
        model, X_val, y_val, "VALIDATION"
    )

    # Test is reserved for final evaluation.
    test_metrics = evaluate_model(
        model, X_test, y_test, "TEST"
    )

    model_path = model_dir / "tuned_xgboost_model.joblib"
    joblib.dump(model, model_path)

    results = {
        "model": "Tuned XGBRegressor",
        "training_records": len(train),
        "parameters": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 1.0,
            "colsample_bytree": 0.8
        },
        "validation": validation_metrics,
        "test": test_metrics
    }

    metrics_path = artifact_dir / "tuned_xgboost_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 55)
    print("TUNED XGBOOST TRAINING COMPLETED")
    print("=" * 55)

    print("Model saved:", model_path)
    print("Metrics saved:", metrics_path)


if __name__ == "__main__":
    train_tuned_xgboost()
