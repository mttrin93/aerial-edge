"""Visual check of the trained model on one val image: what it finds, and what it misses.

    python scripts/predict_example.py                           # default image
    python scripts/predict_example.py --image 0000271_00801_d_0000378 --conf 0.25

Writes to results/:
  prediction_example.png  ground truth (left) next to the model's boxes at conf >= --conf (right)
  prediction_errors.png   every ground-truth box, coloured by what the model did with it:
                          found / found only below --conf / right place but wrong class / missed

A ground-truth box counts as found when a prediction of the same class overlaps it with
IoU >= 0.5, the matching rule of mAP50.
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.plotting import Annotator, colors

from aerial_edge.paths import CONFIGS, DATA_YAML, MODELS, RESULTS

IOU_MATCH = 0.5
# BGR colours for the error map
OUTCOMES = {
    "found": (60, 200, 60),
    "found only at low confidence": (0, 230, 255),
    "right place, wrong class": (0, 150, 255),
    "missed": (40, 40, 230),
}


def load_ground_truth(label: Path, w: int, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    """YOLO-format label file -> (class ids, xyxy boxes in pixels)."""
    g = np.array([line.split() for line in label.read_text().splitlines()], dtype=np.float32)
    x, y, bw, bh = g[:, 1:].T
    boxes = np.stack([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h], 1)
    return torch.from_numpy(g[:, 0]).int(), torch.from_numpy(boxes)


def title_bar(width: int, text: str) -> np.ndarray:
    bar = np.full((50, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    return bar


def classify(gt_cls, gt_box, pred_cls, pred_box, pred_conf, conf: float) -> list[str]:
    """Outcome of every ground-truth box, from the model's predictions at conf >= 0.01."""
    iou = box_iou(gt_box, pred_box)
    outcomes = []
    for i in range(len(gt_box)):
        overlaps = iou[i] >= IOU_MATCH
        same_class = overlaps & (pred_cls == gt_cls[i])
        best = float(pred_conf[same_class].max()) if same_class.any() else 0.0
        if best >= conf:
            outcomes.append("found")
        elif best > 0:
            outcomes.append("found only at low confidence")
        elif overlaps.any():
            outcomes.append("right place, wrong class")
        else:
            outcomes.append("missed")
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--weights", type=Path, default=MODELS / "best.pt")
    parser.add_argument("--image", default="0000271_00801_d_0000378", help="val image stem")
    parser.add_argument("--conf", type=float, default=0.25, help="threshold for drawn boxes")
    args = parser.parse_args()
    max_det = yaml.safe_load((CONFIGS / "export.yaml").read_text())["max_det"]

    image_dir = Path(check_det_dataset(DATA_YAML)["val"])
    image_path = image_dir / f"{args.image}.jpg"
    label_path = image_dir.parent.parent / "labels" / image_dir.name / f"{args.image}.txt"
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]

    model = YOLO(args.weights)
    names = model.names
    # Predict once at a low threshold; --conf only decides what is drawn and what counts as found.
    result = model.predict(image, conf=0.01, max_det=max_det, verbose=False)[0]
    pred_cls, pred_box, pred_conf = result.boxes.cls.int(), result.boxes.xyxy, result.boxes.conf
    gt_cls, gt_box = load_ground_truth(label_path, w, h)

    # 1. Ground truth next to the prediction
    shown = result[result.boxes.conf >= args.conf]
    predicted = shown.plot(line_width=2, font_size=12)
    annotator = Annotator(image.copy(), line_width=2, font_size=12)
    for c, box in zip(gt_cls.tolist(), gt_box.tolist()):
        annotator.box_label(box, names[c], color=colors(c, True))
    left = np.vstack([title_bar(w, f"Ground truth ({len(gt_box)} objects)"), annotator.result()])
    right = np.vstack(
        [title_bar(w, f"Prediction, conf >= {args.conf} ({len(shown)} boxes)"), predicted]
    )
    gap = np.full((h + 50, 20, 3), 255, np.uint8)
    RESULTS.mkdir(exist_ok=True)
    cv2.imwrite(str(RESULTS / "prediction_example.png"), np.hstack([left, gap, right]))

    # 2. Error map: every ground-truth box coloured by its outcome
    outcomes = classify(gt_cls, gt_box, pred_cls, pred_box, pred_conf, args.conf)
    errors = image.copy()
    for box, outcome in zip(gt_box.int().tolist(), outcomes):
        x1, y1, x2, y2 = box
        cv2.rectangle(errors, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), OUTCOMES[outcome], 3)
    legend = np.full((50, w, 3), 255, np.uint8)
    for i, (outcome, colour) in enumerate(OUTCOMES.items()):
        x = 15 + i * w // len(OUTCOMES)
        cv2.rectangle(legend, (x, 15), (x + 25, 38), colour, -1)
        label = outcome.replace("low confidence", f"conf < {args.conf}")
        cv2.putText(legend, label, (x + 32, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(RESULTS / "prediction_errors.png"), np.vstack([legend, errors]))

    print(f"{image_path.name} ({w}x{h}), {len(gt_box)} ground-truth objects:")
    for outcome, n in Counter(outcomes).most_common():
        print(f"  {outcome:30}{n:>4}")
    per_class = Counter((names[int(c)], o) for c, o in zip(gt_cls, outcomes) if o != "found")
    print(f"Not found at conf >= {args.conf}, by class:")
    for (name, outcome), n in sorted(per_class.items()):
        print(f"  {name:16}{outcome:30}{n:>4}")
    print(f"Wrote {RESULTS / 'prediction_example.png'} and {RESULTS / 'prediction_errors.png'}")


if __name__ == "__main__":
    main()
