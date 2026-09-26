"""Preprocessing and evaluation shared by export, quantization, benchmark and evaluation."""

from pathlib import Path

import numpy as np

from aerial_edge.paths import DATA_YAML


def preprocess(image: np.ndarray, imgsz: int) -> np.ndarray:
    """BGR image -> (1, 3, imgsz, imgsz) float32 in [0, 1], letterboxed exactly like Ultralytics."""
    from ultralytics.data.augment import LetterBox

    x = LetterBox((imgsz, imgsz), auto=False)(image=image)[None, ..., ::-1].transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x, dtype=np.float32) / 255


def split_images(split: str) -> list[Path]:
    """Sorted image paths of a VisDrone split ("train", "val" or "test")."""
    from ultralytics.data.utils import check_det_dataset

    return sorted(Path(check_det_dataset(DATA_YAML)[split]).glob("*.jpg"))


def val_metrics(weights: Path, imgsz: int, max_det: int, split: str = "val"):
    """Ultralytics DetMetrics of a .pt or .onnx model, with settings every variant shares.

    batch 1 and rect=False: a static ONNX model only takes square inputs, so PyTorch is run the
    same way. max_det must not be the Ultralytics default (300), which val silently raises.
    """
    from ultralytics import YOLO

    return YOLO(weights, task="detect").val(
        data=DATA_YAML,
        split=split,
        imgsz=imgsz,
        batch=1,
        rect=False,
        max_det=max_det,
        device="cpu",
        plots=False,
        verbose=False,
    )
