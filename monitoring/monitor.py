
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "splits" / "train.parquet"
TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.parquet"
MODEL_PATH = ROOT / "models" / "tuned_xgboost_model.joblib"

REPORT_DIR = ROOT / "monitoring" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "accept_hour",
    "accept_dayofweek",
    "accept_month",
    "distance_km",
    "lat",
    "lng"
]

CATEGORICAL_FEATURES = [
    "city",
    "region_id",
    "aoi_type"
]


def calculate_psi(reference, current, bins=10):
    """Calculate Population Stability Index for two numeric distributions."""

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if reference.empty or current.empty:
        return None

    breakpoints = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(breakpoints) < 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    epsilon = 0.0001
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    psi = np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    )

    return float(psi)


def calculate_categorical_psi(reference, current):
    """Calculate PSI for categorical features."""

    reference = reference.fillna("MISSING").astype(str)
    current = current.fillna("MISSING").astype(str)

    categories = set(reference.unique()) | set(current.unique())

    ref_pct = reference.value_counts(normalize=True)
    cur_pct = current.value_counts(normalize=True)

    ref_values = np.array([ref_pct.get(c, 0) for c in categories])
    cur_values = np.array([cur_pct.get(c, 0) for c in categories])

    epsilon = 0.0001
    ref_values = np.clip(ref_values, epsilon, None)
    cur_values = np.clip(cur_values, epsilon, None)

    psi = np.sum(
        (cur_values - ref_values) * np.log(cur_values / ref_values)
    )

    return float(psi)


def get_drift_status(psi):
    if psi is None:
        return "NOT_AVAILABLE"
    elif psi < 0.10:
        return "NO_SIGNIFICANT_DRIFT"
    elif psi < 0.25:
        return "MODERATE_DRIFT"
    else:
        return "SIGNIFICANT_DRIFT"


def main():

    print("Loading reference and current datasets...")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Use a sample for faster monitoring
    reference = train_df.sample(
        n=min(100000, len(train_df)), random_state=42
    )

    current = test_df.sample(
        n=min(100000, len(test_df)), random_state=42
    )

    results = []

    print("\nChecking numerical features...")

    for feature in NUMERIC_FEATURES:
        psi = calculate_psi(reference[feature], current[feature])

        results.append({
            "feature": feature,
            "type": "numerical",
            "psi": psi,
            "status": get_drift_status(psi)
        })

    print("Checking categorical features...")

    for feature in CATEGORICAL_FEATURES:
        psi = calculate_categorical_psi(
            reference[feature], current[feature]
        )

        results.append({
            "feature": feature,
            "type": "categorical",
            "psi": psi,
            "status": get_drift_status(psi)
        })

    report = pd.DataFrame(results)

    print("\nDATA DRIFT REPORT")
    print("=" * 65)
    print(report.to_string(index=False))

    report.to_csv(REPORT_DIR / "drift_report.csv", index=False)

    # Evaluate model performance on the October test batch
    print("\nEvaluating model performance...")

    model = joblib.load(MODEL_PATH)

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_test = test_df[feature_columns]
    y_test = test_df["target_duration"]

    predictions = model.predict(X_test)

    performance = {
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "test_r2": float(r2_score(y_test, predictions))
    }

    with open(REPORT_DIR / "performance_report.json", "w") as file:
        json.dump(performance, file, indent=4)

    print("\nMODEL PERFORMANCE")
    print("=" * 40)
    print(json.dumps(performance, indent=4))

    print("\nMonitoring completed!")
    print("Reports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
