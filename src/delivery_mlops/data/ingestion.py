
from pathlib import Path
from huggingface_hub import snapshot_download


def download_dataset():

    # Project root directory
    project_root = Path(__file__).resolve().parents[3]

    # Dataset destination
    raw_data_dir = project_root / "data" / "raw" / "lade-d"

    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print("Starting LaDe-D dataset download...")

    # Official Hugging Face repository
    repo_id = "Cainiao-AI/LaDe-D"

    # Download all five city files
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_dir=str(raw_data_dir)
    )

    data_dir = Path(downloaded_path) / "data"

    print("\nDataset downloaded successfully!")

    # Verify downloaded files
    parquet_files = sorted(data_dir.glob("*.parquet"))

    for file in parquet_files:
        size_mb = file.stat().st_size / (1024 * 1024)

        print(
            f"{file.name} | {size_mb:.2f} MB"
        )

    if len(parquet_files) != 5:
        raise RuntimeError(
            f"Expected 5 city files, found {len(parquet_files)}"
        )

    print("\nTotal city files:", len(parquet_files))

    return data_dir


if __name__ == "__main__":
    download_dataset()
