
from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = ROOT / "data" / "incoming" / "incoming_batch.parquet"
REPORT_FILE = ROOT / "artifacts" / "quality_monitoring" / "incoming_validation.json"

FEATURES = [
    "city", "region_id", "aoi_type",
    "accept_hour", "accept_dayofweek",
    "accept_month", "distance_km",
    "lat", "lng"
]

TARGET = "target_duration"
REQUIRED = FEATURES + [TARGET]

def validate_data():
    df = pd.read_parquet(INPUT_FILE)

    missing_columns = [
        c for c in REQUIRED if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    missing_values = df[REQUIRED].isnull().sum()

    duplicate_records = int(df.duplicated().sum())

    invalid_targets = int(
        (
            (df[TARGET] < 0) |
            (df[TARGET] > 10080) |
            (~df[TARGET].map(
                lambda x: pd.notna(x)
            ))
        ).sum()
    )

    report = {
        "dataset_size": int(len(df)),
        "required_columns": REQUIRED,
        "missing_values": {
            k: int(v) for k, v in missing_values.items()
        },
        "missing_feature_values_handled_by_imputer": int(
            missing_values[FEATURES].sum()
        ),
        "duplicate_records": duplicate_records,
        "invalid_target_records": invalid_targets,
        "target_summary": {
            "mean": float(df[TARGET].mean()),
            "median": float(df[TARGET].median()),
            "min": float(df[TARGET].min()),
            "max": float(df[TARGET].max())
        },
        "validation_passed": bool(
            len(missing_columns) == 0
            and missing_values[TARGET] == 0
            and duplicate_records == 0
            and invalid_targets == 0
        )
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.write_text(
        json.dumps(report, indent=4)
    )

    print(json.dumps(report, indent=4))

    if not report["validation_passed"]:
        raise ValueError("Incoming data validation failed.")

    print("\nVALIDATION PASSED!")

if __name__ == "__main__":
    validate_data()
