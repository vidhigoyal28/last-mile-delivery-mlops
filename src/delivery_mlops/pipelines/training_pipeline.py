
from pathlib import Path
import pandas as pd

def investigate_records():
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "processed"

    for file in sorted(data_dir.glob("features_*.parquet")):
        df = pd.read_parquet(file)

        print("\n" + "=" * 65)
        print(file.name)
        print("=" * 65)

        # Records after October 31, 2022
        late_dates = df[
            df["accept_datetime"] > pd.Timestamp("2022-10-31 23:59:59")
        ]

        if not late_dates.empty:
            print("\nRecords after October 31:")
            print(
                late_dates[
                    ["city", "accept_datetime", "target_duration"]
                ].to_string(index=False)
            )
        else:
            print("\nNo records after October 31.")

        # Longest delivery durations
        print("\nTop 10 longest delivery durations:")
        print(
            df.nlargest(10, "target_duration")[
                ["city", "accept_datetime", "target_duration"]
            ].to_string(index=False)
        )

        # Tasks longer than 7 days
        extreme = df[df["target_duration"] > 10080]

        print("\nTasks longer than 7 days:", len(extreme))

        # Zero-duration records
        zero = df[df["target_duration"] == 0]

        print("Zero-duration records:", len(zero))

        if not zero.empty:
            print("Sample zero-duration records:")
            print(
                zero[
                    ["city", "accept_datetime", "target_duration"]
                ].head(5).to_string(index=False)
            )


if __name__ == "__main__":
    investigate_records()
