
from pathlib import Path
import pyarrow.parquet as pq


EXPECTED_COLUMNS = {
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
}


def validate_dataset():

    project_root = Path(__file__).resolve().parents[3]

    data_dir = (
        project_root / "data" / "raw" / "lade-d" / "data"
    )

    parquet_files = sorted(
        data_dir.glob("delivery_*.parquet")
    )

    if len(parquet_files) != 5:
        raise ValueError(
            f"Expected 5 files, found {len(parquet_files)}"
        )

    total_records = 0

    print("\nSTARTING DATA VALIDATION")
    print("=" * 50)

    for file in parquet_files:

        parquet_file = pq.ParquetFile(file)

        schema = parquet_file.schema_arrow
        columns = set(schema.names)

        # Validate columns
        missing_columns = EXPECTED_COLUMNS - columns

        if missing_columns:
            raise ValueError(
                f"{file.name}: Missing columns {missing_columns}"
            )

        # Validate record count
        records = parquet_file.metadata.num_rows

        total_records += records

        print(f"\nFile: {file.name}")
        print(f"Records: {records:,}")
        print(f"Columns: {len(columns)}")

        print("Schema validation: PASSED")

        # Count missing values in batches
        null_counts = {
            column: 0 for column in EXPECTED_COLUMNS
        }

        for batch in parquet_file.iter_batches(
            batch_size=100000
        ):

            for column in EXPECTED_COLUMNS:
                null_counts[column] += (
                    batch.column(
                        batch.schema.get_field_index(column)
                    ).null_count
                )

        missing = {
            col: count
            for col, count in null_counts.items()
            if count > 0
        }

        print("Missing values:", missing if missing else "None")

    print("\n" + "=" * 50)
    print("DATA VALIDATION COMPLETED")
    print("Total records:", f"{total_records:,}")
    print("Total city files:", len(parquet_files))


if __name__ == "__main__":
    validate_dataset()
