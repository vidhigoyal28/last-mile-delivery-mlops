
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def evaluate_model(model, X, y, name):
    predictions = model.predict(X)

    metrics = {
        "MAE": float(mean_absolute_error(y, predictions)),
        "RMSE": float(mean_squared_error(y, predictions) ** 0.5),
        "R2": float(r2_score(y, predictions))
    }

    print(f"\n{name} RESULTS")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics


def train_random_forest():
    root = Path(__file__).resolve().parents[3]
    split_dir = root / "data" / "processed" / "splits"

    model_dir = root / "models"
    artifact_dir = root / "artifacts"
    model_dir.mkdir(exist_ok=True)
    artifact_dir.mkdir(exist_ok=True)

    print("Loading datasets...")
    train = pd.read_parquet(split_dir / "train.parquet")
    val = pd.read_parquet(split_dir / "validation.parquet")
    test = pd.read_parquet(split_dir / "test.parquet")

    target = "target_duration"
    excluded = [target, "accept_datetime"]
    features = [c for c in train.columns if c not in excluded]

    categorical = ["city", "region_id", "aoi_type"]
    numerical = [c for c in features if c not in categorical]

    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test, y_test = test[features], test[target]

    # Preprocessing: impute missing values and encode categories.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1
                    ))
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
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=10,
            max_features=0.7,
            n_jobs=-1,
            random_state=42
        ))
    ])

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    print("Training completed!")

    validation_metrics = evaluate_model(
        model, X_val, y_val, "VALIDATION"
    )

    test_metrics = evaluate_model(
        model, X_test, y_test, "TEST"
    )

    joblib.dump(model, model_dir / "random_forest_model.joblib")

    results = {
        "model": "RandomForestRegressor",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_leaf": 10,
            "max_features": 0.7
        },
        "validation": validation_metrics,
        "test": test_metrics
    }

    with open(artifact_dir / "random_forest_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nRandom Forest model and metrics saved.")


if __name__ == "__main__":
    train_random_forest()
