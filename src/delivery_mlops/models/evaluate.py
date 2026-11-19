
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_by_city():
    root = Path(__file__).resolve().parents[3]

    test_file = root / "data" / "processed" / "splits" / "test.parquet"
    model_file = root / "models" / "xgboost_model.joblib"
    output_file = root / "artifacts" / "xgboost_city_metrics.csv"

    print("Loading test data and XGBoost model...")

    test = pd.read_parquet(test_file)
    model = joblib.load(model_file)

    target = "target_duration"
    excluded = [target, "accept_datetime"]

    features = [
        col for col in test.columns
        if col not in excluded
    ]

    print("Generating predictions...")
    predictions = model.predict(test[features])

    test = test.copy()
    test["predicted_duration"] = predictions

    results = []

    for city, city_data in test.groupby("city"):
        actual = city_data[target]
        predicted = city_data["predicted_duration"]

        results.append({
            "city": city,
            "records": len(city_data),
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": mean_squared_error(actual, predicted) ** 0.5,
            "R2": r2_score(actual, predicted)
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(output_file, index=False)

    print("\nCITY-WISE XGBOOST PERFORMANCE")
    print("=" * 65)
    print(results_df.to_string(index=False))

    print("\nSaved metrics to:", output_file)


if __name__ == "__main__":
    evaluate_by_city()
