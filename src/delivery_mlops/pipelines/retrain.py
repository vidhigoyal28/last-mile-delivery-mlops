
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

DRIFT_REPORT = ROOT / "monitoring" / "reports" / "drift_report.csv"
TRAIN_SCRIPT = ROOT / "src" / "delivery_mlops" / "models" / "train_tuned_xgboost.py"

DRIFT_THRESHOLD = 0.25


def main():
    print("=" * 55)
    print("AUTOMATED MODEL RETRAINING PIPELINE")
    print("=" * 55)

    if not DRIFT_REPORT.exists():
        print("Drift report not found. Run monitoring first.")
        return

    report = pd.read_csv(DRIFT_REPORT)

    significant_drift = report[
        (report["psi"] > DRIFT_THRESHOLD)
    ]

    print("\nFeatures with significant drift:")
    print(significant_drift[["feature", "psi"]].to_string(index=False))

    if significant_drift.empty:
        print("\nNo significant drift detected.")
        print("Retraining is not required.")
        return

    print("\nSignificant drift detected!")
    print("Starting model retraining...")

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=ROOT
    )

    if result.returncode == 0:
        print("\nModel retraining completed successfully.")
        print("Updated model and metrics saved.")
    else:
        print("\nRetraining failed. Check the training logs.")


if __name__ == "__main__":
    main()
