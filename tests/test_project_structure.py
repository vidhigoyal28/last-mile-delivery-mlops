
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_file_exists():
    assert (ROOT / "requirements.txt").exists()


def test_api_file_exists():
    assert (ROOT / "api" / "main.py").exists()


def test_dockerfile_exists():
    assert (ROOT / "Dockerfile").exists()


def test_training_pipeline_exists():
    assert (
        ROOT / "src" / "delivery_mlops" / "pipelines" / "training_pipeline.py"
    ).exists()
