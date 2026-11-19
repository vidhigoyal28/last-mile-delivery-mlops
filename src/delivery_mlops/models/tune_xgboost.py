
from pathlib import Path
import json

import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error


def tune_xgboost():

    root = Path(__file__).resolve().parents[3]
    split_dir = root / "data" / "processed" / "splits"
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(exist_ok=True)

    print("Loading training and validation datasets...")

    train = pd.read_parquet(split_dir / "train.parquet")
    val = pd.read_parquet(split_dir / "validation.parquet")

    # Sample 300,000 training records
    train_sample = train.sample(
        n=min(300000, len(train)),
        random_state=42
    )

    print("Training sample:", len(train_sample))
    print("Validation records:", len(val))

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

    X_train = train_sample[features]
    y_train = train_sample[target]

    X_val = val[features]
    y_val = val[target]

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

    print("\nTransforming training and validation features...")

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    parameter_configs = [
        {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.03,
            "min_child_weight": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 1.0,
            "colsample_bytree": 0.8
        }
    ]

    results = []

    for index, params in enumerate(parameter_configs, start=1):

        print(f"\nExperiment {index}/{len(parameter_configs)}")
        print("Parameters:", params)

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=4,
            random_state=42,
            **params
        )

        model.fit(
            X_train_transformed,
            y_train,
            verbose=False
        )

        predictions = model.predict(X_val_transformed)

        mae = mean_absolute_error(y_val, predictions)

        print(f"Validation MAE: {mae:.4f}")

        results.append({
            "experiment": index,
            "validation_MAE": float(mae),
            "parameters": params
        })

    results.sort(key=lambda item: item["validation_MAE"])

    best_result = results[0]

    print("\n" + "=" * 55)
    print("BEST CONFIGURATION")
    print("=" * 55)

    print(json.dumps(best_result, indent=4))

    with open(
        artifact_dir / "xgboost_tuning_results.json",
        "w"
    ) as file:
        json.dump({
            "experiments": results,
            "best_configuration": best_result
        }, file, indent=4)

    print("\nTuning results saved successfully!")


if __name__ == "__main__":
    tune_xgboost()
