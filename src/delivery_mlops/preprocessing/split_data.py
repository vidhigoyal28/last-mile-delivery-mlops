
from pathlib import Path
import pandas as pd

def split_dataset():
    project_root = Path(__file__).resolve().parents[3]

    input_file = (
        project_root / "data" / "interim" / "model_dataset.parquet"
    )

    output_dir = project_root / "data" / "processed" / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_file)

    # Chronological split
    train = df[
        df["accept_datetime"] < "2022-09-01"
    ].copy()

    validation = df[
        (df["accept_datetime"] >= "2022-09-01") &
        (df["accept_datetime"] < "2022-10-01")
    ].copy()

    test = df[
        (df["accept_datetime"] >= "2022-10-01") &
        (df["accept_datetime"] < "2022-11-01")
    ].copy()

    datasets = {
        "train": train,
        "validation": validation,
        "test": test
    }

    print("\nCHRONOLOGICAL DATA SPLIT")
    print("=" * 55)

    for name, data in datasets.items():
        output_file = output_dir / f"{name}.parquet"
        data.to_parquet(output_file, index=False)

        print(f"\n{name.upper()}")
        print(f"Records: {len(data):,}")
        print(f"Date range: {data['accept_datetime'].min()} to "
              f"{data['accept_datetime'].max()}")
        print("\nRecords by city:")
        print(data["city"].value_counts())
        print(f"Saved: {output_file}")

    print("\nTOTAL SPLIT RECORDS:",
          sum(len(data) for data in datasets.values()))

    assert len(train) + len(validation) + len(test) == len(df)


if __name__ == "__main__":
    split_dataset()
