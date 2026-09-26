"""Step 3: train YOLO on VisDrone with the settings in configs/train.yaml.

Examples:
    python scripts/train.py                            # full run, as in the config
    python scripts/train.py epochs=1 fraction=0.01     # smoke test: key=value overrides the config
    python scripts/train.py --resume runs/visdrone/yolo26n_640/weights/last.pt

MLflow: Ultralytics logs params, per-epoch metrics and weights to $MLFLOW_TRACKING_URI
(default: the git-ignored SQLite DB mlflow/mlflow.db). With --backup-mlflow SRC DEST, the SQLite tracking DB is copied from SRC to DEST at every epoch
so the record survives a Colab disconnect (SQLite itself should not live on Google Drive).
"""

import argparse
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

from aerial_edge.paths import CONFIGS, ROOT

EXPERIMENT = "aerial-edge-visdrone"


def parse_overrides(pairs: list[str]) -> dict:
    """Turn ["epochs=1", "fraction=0.01"] into {"epochs": 1, "fraction": 0.01}."""
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"Override '{pair}' must look like key=value")
        overrides[key] = yaml.safe_load(value)
    return overrides


def backup_sqlite(src: Path, dest: Path) -> None:
    """Consistent copy of a live SQLite DB (the backup API never copies a half-written page)."""
    if not src.exists():
        return
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / src.name
        with sqlite3.connect(src) as source, sqlite3.connect(snapshot) as target:
            source.backup(target)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(".partial")
        shutil.copyfile(snapshot, partial)
        os.replace(partial, dest)  # atomic: DEST is either the old or the new copy, never half


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("overrides", nargs="*", help="key=value pairs that override the config")
    parser.add_argument("--config", type=Path, default=CONFIGS / "train.yaml")
    parser.add_argument("--resume", type=Path, help="last.pt of an interrupted run")
    parser.add_argument("--backup-mlflow", nargs=2, type=Path, metavar=("SRC", "DEST"))
    args = parser.parse_args()

    SETTINGS.update({"mlflow": True})
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", EXPERIMENT)
    if "MLFLOW_TRACKING_URI" not in os.environ:
        # Otherwise Ultralytics falls back to a file store under runs/, which MLflow 3 deprecates.
        (ROOT / "mlflow").mkdir(exist_ok=True)
        os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{ROOT / 'mlflow' / 'mlflow.db'}"
    print(f"MLflow tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")

    if args.resume:
        model = YOLO(args.resume)
        train_kwargs = {"resume": True}  # every setting comes from the checkpoint
    else:
        cfg = yaml.safe_load(args.config.read_text()) | parse_overrides(args.overrides)
        model = YOLO(cfg.pop("model"))
        train_kwargs = cfg

    if args.backup_mlflow:
        src, dest = args.backup_mlflow
        # Our callbacks run before the MLflow ones at the same event, so back up at the start of
        # each epoch: that snapshot holds everything logged for the previous epoch.
        model.add_callback("on_train_epoch_start", lambda trainer: backup_sqlite(src, dest))

    model.train(**train_kwargs)

    if args.backup_mlflow:
        backup_sqlite(*args.backup_mlflow)
    trainer = model.trainer
    print(f"\nBest weights: {trainer.best}")
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        print(f"  val {key}: {trainer.metrics.get(key, float('nan')):.4f}")


if __name__ == "__main__":
    main()
