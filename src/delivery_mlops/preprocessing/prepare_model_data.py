
from pathlib import Path
import pandas as pd

def prepare_model_data():
    project_root = Path(__file__).resolve().parents[3]
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "data" / "interim"

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(processed_dir.glob("features_*.parquet"))

    if len(files) != 5:
        raise FileNotFoundError(
            f"Expected 5 city files, found {len(files)}"
        )

    dataframes = []

    for file in files:
        df = pd.read_parquet(file)
        print(f"Loaded {file.name}: {len(df):,} records")
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)

    initial_count = len(data)

    # Keep a common evaluation period across all cities
    date_mask = (
        (data["accept_datetime"] >= "2022-05-01") &
        (data["accept_datetime"] < "2022-11-01")
    )

    data = data.loc[date_mask].copy()
    after_date_filter = len(data)

    # Keep nonnegative durations up to 7 days.
    # Zero-duration records are intentionally retained.
    duration_mask = (
        (data["target_duration"] >= 0) &
        (data["target_duration"] <= 10080)
    )

    data = data.loc[duration_mask].copy()
    after_duration_filter = len(data)

    # Sort chronologically
    data = data.sort_values("accept_datetime")

    output_file = output_dir / "model_dataset.parquet"
    data.to_parquet(output_file, index=False)

    print("\n" + "=" * 55)
    print("MODELLING DATASET PREPARATION")
    print("=" * 55)

    print(f"Initial records: {initial_count:,}")
    print(f"After date filter: {after_date_filter:,}")
    print(f"After duration filter: {after_duration_filter:,}")
    print(f"Excluded records: {initial_count - after_duration_filter:,}")

    print("\nRecords by city:")
    print(data["city"].value_counts())

    print("\nTarget statistics:")
    print(data["target_duration"].describe())

    print("\nDate range:")
    print(data["accept_datetime"].min())
    print(data["accept_datetime"].max())

    print(f"\nSaved dataset to: {output_file}")


if __name__ == "__main__":
    prepare_model_data()
