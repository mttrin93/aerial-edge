"""Canonical locations for data and artifacts, so every script agrees on them."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATASETS = ROOT / "datasets"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Ultralytics ships a VisDrone.yaml that downloads and converts the dataset itself.
DATA_YAML = "VisDrone.yaml"
IMGSZ = 640
SEED = 0
