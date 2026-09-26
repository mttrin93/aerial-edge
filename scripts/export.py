"""Step 4: export best.pt to ONNX and check that the ONNX model matches PyTorch.

    python scripts/export.py                  # export, parity check, val mAP of both
    python scripts/export.py --skip-val       # export and parity check only (fast)

Two checks, both with the settings in configs/export.yaml:
  1. Parity: the same letterboxed val images through PyTorch and ONNX Runtime; reports the max
     absolute difference of the raw (4 + nc, 8400) outputs: box xywh in pixels, class scores.
     With the default nms=None, YOLO26 exports its one-to-many head and NMS runs afterwards in
     Python, the same way the Colab val was measured.
  2. Val mAP of both models, with identical settings (batch 1, square 640 input, fixed max_det).
     ONNX runs on square inputs only, so PyTorch is validated with rect=False as well.

Writes models/best.onnx and results/export_parity.csv.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import check_det_dataset

from aerial_edge.paths import CONFIGS, DATA_YAML, MODELS, RESULTS

PARITY_IMAGES = 16


def letterbox_batch(paths: list[Path], imgsz: int) -> np.ndarray:
    """Preprocess images like Ultralytics: letterbox to imgsz, BGR->RGB, CHW, float in [0, 1]."""
    import cv2

    letterbox = LetterBox((imgsz, imgsz), auto=False)
    images = [letterbox(image=cv2.imread(str(p))) for p in paths]
    x = np.stack(images)[..., ::-1].transpose(0, 3, 1, 2)
    return np.ascontiguousarray(x, dtype=np.float32) / 255


def parity(pt_path: Path, onnx_path: Path, cfg: dict) -> dict:
    """Max abs difference between PyTorch and ONNX Runtime outputs on a few val images."""
    val_dir = Path(check_det_dataset(DATA_YAML)["val"])
    paths = sorted(val_dir.glob("*.jpg"))[:PARITY_IMAGES]

    model = YOLO(pt_path).model.fuse().eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    box_diff = score_diff = 0.0
    for path in paths:
        x = letterbox_batch([path], cfg["imgsz"])
        with torch.no_grad():
            y_pt = model(torch.from_numpy(x))
        y_pt = (y_pt[0] if isinstance(y_pt, (list, tuple)) else y_pt).numpy()
        y_ort = session.run(None, {input_name: x})[0]
        assert y_pt.shape == y_ort.shape, f"shape mismatch: {y_pt.shape} vs {y_ort.shape}"
        box_diff = max(box_diff, float(np.abs(y_pt[:, :4] - y_ort[:, :4]).max()))
        score_diff = max(score_diff, float(np.abs(y_pt[:, 4:] - y_ort[:, 4:]).max()))

    return {
        "images": len(paths),
        "max_abs_diff_box_px": box_diff,
        "max_abs_diff_score": score_diff,
    }


def val_map(weights: Path, cfg: dict) -> tuple[float, float]:
    metrics = YOLO(weights, task="detect").val(
        data=DATA_YAML,
        split="val",
        imgsz=cfg["imgsz"],
        batch=1,
        rect=False,
        max_det=cfg["max_det"],
        device="cpu",
        plots=False,
        verbose=False,
    )
    return metrics.box.map50, metrics.box.map


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--weights", type=Path, default=MODELS / "best.pt")
    parser.add_argument("--config", type=Path, default=CONFIGS / "export.yaml")
    parser.add_argument("--skip-val", action="store_true", help="skip the (slow) val mAP runs")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    onnx_path = Path(
        YOLO(args.weights).export(
            format="onnx",
            imgsz=cfg["imgsz"],
            opset=cfg["opset"],
            dynamic=cfg["dynamic"],
            simplify=cfg["simplify"],
            max_det=cfg["max_det"],
            device="cpu",
        )
    )
    print(f"\nONNX model: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    check = parity(args.weights, onnx_path, cfg)
    print(f"\nParity on {check['images']} val images (raw outputs, before NMS):")
    print(f"  max abs diff, box (px):  {check['max_abs_diff_box_px']:.2e}")
    print(f"  max abs diff, score:     {check['max_abs_diff_score']:.2e}")

    rows = []
    for runtime, path in (("PyTorch", args.weights), ("ONNX Runtime", onnx_path)):
        row = {
            "variant": "FP32",
            "runtime": runtime,
            "size_mb": round(path.stat().st_size / 1e6, 2),
        }
        if not args.skip_val:
            map50, map50_95 = val_map(path, cfg)
            row |= {"map50": round(map50, 4), "map50_95": round(map50_95, 4)}
        rows.append(row)

    print()
    for row in rows:
        print("  " + ", ".join(f"{k}={v}" for k, v in row.items()))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "export_parity.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[*rows[0], *check])
        writer.writeheader()
        for row in rows:
            writer.writerow(row | check)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
