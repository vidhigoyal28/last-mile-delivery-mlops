
from pathlib import Path
import pandas as pd


# Columns required for our first ML experiment
REQUIRED_COLUMNS = [
    "order_id",
    "region_id",
    "city",
    "courier_id",
    "lng",
    "lat",
    "aoi_id",
    "aoi_type",
    "accept_time",
    "accept_gps_time",
    "accept_gps_lng",
    "accept_gps_lat",
    "delivery_time",
    "delivery_gps_time",
    "delivery_gps_lng",
    "delivery_gps_lat",
    "ds"
]


def get_project_root():
    """Find the main project directory."""
    return Path(__file__).resolve().parents[3]


def preprocess_city(file_path, output_dir):
    """
    Preprocess one city's delivery records.
    """

    print(f"\nProcessing: {file_path.name}")

    # Read one city file
    df = pd.read_parquet(file_path)

    # Verify required columns
    missing_columns = (
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Keep required columns only
    df = df[REQUIRED_COLUMNS].copy()

    # Convert timestamps into datetime objects.
    # Dataset observation year: 2022.
    df["accept_datetime"] = pd.to_datetime(
        "2022-" + df["accept_time"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    df["delivery_datetime"] = pd.to_datetime(
        "2022-" + df["delivery_time"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # Calculate task duration in minutes
    df["delivery_duration_min"] = (
        df["delivery_datetime"] -
        df["accept_datetime"]
    ).dt.total_seconds() / 60

    # Identify valid timestamps and durations
    df["timestamp_valid"] = (
        df["accept_datetime"].notna() &
        df["delivery_datetime"].notna()
    )

    df["duration_valid"] = (
        df["timestamp_valid"] &
        df["delivery_duration_min"].ge(0)
    )

    # Extract features from acceptance timestamp
    df["accept_hour"] = (
        df["accept_datetime"].dt.hour
    )

    df["accept_dayofweek"] = (
        df["accept_datetime"].dt.dayofweek
    )

    df["accept_month"] = (
        df["accept_datetime"].dt.month
    )

    # Create ML target.
    # Invalid durations are represented as missing.
    df["target_duration"] = (
        df["delivery_duration_min"].where(
            df["duration_valid"]
        )
    )

    # Report preprocessing statistics
    total_records = len(df)
    valid_records = int(df["duration_valid"].sum())
    invalid_records = total_records - valid_records

    print("Total records:", f"{total_records:,}")
    print("Valid durations:", f"{valid_records:,}")
    print("Invalid durations:", f"{invalid_records:,}")

    print(
        "Missing acceptance GPS:",
        df["accept_gps_lat"].isna().sum()
    )

    # Save processed city data
    output_file = (
        output_dir /
        f"preprocessed_{file_path.stem.split('-')[0].replace('delivery_', '')}.parquet"
    )

    df.to_parquet(
        output_file,
        index=False,
        compression="snappy"
    )

    print("Saved:", output_file.name)

    return {
        "city_file": file_path.name,
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "output_file": str(output_file)
    }


def run_preprocessing():

    project_root = get_project_root()

    raw_dir = (
        project_root / "data" / "raw" /
        "lade-d" / "data"
    )

    output_dir = (
        project_root / "data" / "interim"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    parquet_files = sorted(
        raw_dir.glob("delivery_*.parquet")
    )

    if len(parquet_files) != 5:
        raise FileNotFoundError(
            f"Expected 5 city files, found {len(parquet_files)}"
        )

    print("=" * 55)
    print("STARTING DATA PREPROCESSING")
    print("=" * 55)

    results = []

    for file_path in parquet_files:

        result = preprocess_city(
            file_path,
            output_dir
        )

        results.append(result)

    # Display overall summary
    summary = pd.DataFrame(results)

    print("\nPREPROCESSING SUMMARY")
    print(summary.to_string(index=False))

    print("\nTotal records processed:",
          f"{summary['total_records'].sum():,}")

    print("Total valid records:",
          f"{summary['valid_records'].sum():,}")

    print("Total invalid records:",
          f"{summary['invalid_records'].sum():,}")

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_preprocessing()
