
from pathlib import Path
import numpy as np
import pandas as pd


def get_project_root():
    return Path(__file__).resolve().parents[3]


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance in kilometers.
    """

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 6371 * 2 * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


def build_features(df):

    df = df.copy()

    # Convert categorical columns to strings
    categorical_columns = [
        "city",
        "region_id",
        "aoi_type"
    ]

    for col in categorical_columns:
        df[col] = (
            df[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

    # Time-based features
    df["accept_hour"] = (
        df["accept_datetime"].dt.hour
    )

    df["accept_dayofweek"] = (
        df["accept_datetime"].dt.dayofweek
    )

    df["accept_month"] = (
        df["accept_datetime"].dt.month
    )

    # Validate GPS coordinates
    gps_valid = (
        df["accept_gps_lat"].between(-90, 90)
        & df["accept_gps_lng"].between(-180, 180)
        & df["lat"].between(-90, 90)
        & df["lng"].between(-180, 180)
    )

    # Calculate distance only when all coordinates exist
    df["distance_km"] = np.nan

    df.loc[gps_valid, "distance_km"] = (
        calculate_distance(
            df.loc[gps_valid, "accept_gps_lat"],
            df.loc[gps_valid, "accept_gps_lng"],
            df.loc[gps_valid, "lat"],
            df.loc[gps_valid, "lng"]
        )
    )

    # Select model features
    feature_columns = [
        "city",
        "region_id",
        "aoi_type",
        "accept_datetime",
        "accept_hour",
        "accept_dayofweek",
        "accept_month",
        "distance_km",
        "lat",
        "lng"
    ]

    # Keep only valid target records
    ml_data = df.loc[
        df["duration_valid"]
        & df["accept_datetime"].notna(),
        feature_columns + ["target_duration"]
    ].copy()

    # Check target
    ml_data = ml_data[
        np.isfinite(ml_data["target_duration"])
        & (ml_data["target_duration"] >= 0)
    ]

    return ml_data


def run_feature_engineering():

    project_root = get_project_root()

    interim_dir = project_root / "data" / "interim"

    output_dir = project_root / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        interim_dir.glob("preprocessed_*.parquet")
    )

    if len(files) != 5:
        raise FileNotFoundError(
            f"Expected 5 processed files, found {len(files)}"
        )

    total_records = 0

    print("=" * 55)
    print("STARTING FEATURE ENGINEERING")
    print("=" * 55)

    for file in files:

        print(f"\nProcessing {file.name}")

        df = pd.read_parquet(file)

        ml_data = build_features(df)

        output_file = (
            output_dir / f"features_{file.stem.replace('preprocessed_', '')}.parquet"
        )

        ml_data.to_parquet(
            output_file,
            index=False,
            compression="snappy"
        )

        total_records += len(ml_data)

        print("Input records:", f"{len(df):,}")
        print("Output records:", f"{len(ml_data):,}")
        print("Features:", len(ml_data.columns) - 1)
        print("Saved:", output_file.name)

    print("\n" + "=" * 55)
    print("FEATURE ENGINEERING COMPLETED")
    print("Total ML records:", f"{total_records:,}")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    run_feature_engineering()
