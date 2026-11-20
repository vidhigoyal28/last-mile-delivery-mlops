
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = ROOT / "data/incoming/incoming_batch.parquet"

TRAIN_FILE = ROOT / "data/incoming/incoming_train.parquet"
HOLDOUT_FILE = ROOT / "data/incoming/incoming_holdout.parquet"

def prepare_split():

    df = pd.read_parquet(INPUT_FILE)

    incoming_train, incoming_holdout = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

    incoming_train.to_parquet(TRAIN_FILE, index=False)
    incoming_holdout.to_parquet(HOLDOUT_FILE, index=False)

    print("Incoming data split completed!")
    print("Candidate training records:", len(incoming_train))
    print("Evaluation holdout records:", len(incoming_holdout))

    print("\nFiles saved:")
    print(TRAIN_FILE)
    print(HOLDOUT_FILE)

if __name__ == "__main__":
    prepare_split()

