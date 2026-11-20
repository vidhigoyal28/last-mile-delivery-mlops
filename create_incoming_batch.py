
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

TEST_FILE = ROOT / "data" / "processed" / "splits" / "test.parquet"
OUTPUT_FILE = ROOT / "data" / "incoming" / "incoming_batch.parquet"

SAMPLE_SIZE = 100_000
RANDOM_STATE = 42

def create_batch():
    print("Loading October test dataset...")

    df = pd.read_parquet(TEST_FILE)

    print("Original dataset:", df.shape)

    if len(df) < SAMPLE_SIZE:
        raise ValueError("Not enough records for incoming batch.")

    batch = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    batch.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nIncoming batch created successfully!")
    print("Records:", len(batch))
    print("Saved at:", OUTPUT_FILE)
    print("\nTarget summary:")
    print(batch["target_duration"].describe())

if __name__ == "__main__":
    create_batch()
