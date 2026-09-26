"""Step 6: val mAP of every model variant, overall and per class.

    python scripts/evaluate.py                                  # FP32 ONNX + the quantized ones
    python scripts/evaluate.py models/best_int8_static.onnx

All variants use the same settings (batch 1, square 640 input, max_det from configs/export.yaml,
CPU), so differences come from the model alone. Per-class mAP matters: the EDA predicts that rare
and tiny classes lose accuracy first. Writes results/evaluate.csv; the reported delta is against
the first model given (FP32 ONNX by default).
"""

import argparse
import csv
from pathlib import Path

import yaml
from tabulate import tabulate

from aerial_edge.inference import val_metrics
from aerial_edge.paths import CONFIGS, MODELS, RESULTS

DEFAULT_MODELS = [
    MODELS / f"best{tag}.onnx" for tag in ("", "_fp16", "_int8_dynamic", "_int8_static")
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("models", nargs="*", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--config", type=Path, default=CONFIGS / "export.yaml")
    parser.add_argument("--split", default="val", help="test only once, at the very end")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    rows = []
    for path in args.models:
        print(f"evaluating {path.name} on {args.split} ...", flush=True)
        metrics = val_metrics(path, cfg["imgsz"], cfg["max_det"], args.split)
        box = metrics.box
        row = {
            "file": path.name,
            "split": args.split,
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "map50": round(box.map50, 4),
            "map50_95": round(box.map, 4),
        }
        # box.maps: mAP50-95 per class id
        row |= {f"map50_95_{name}": round(box.maps[i], 4) for i, name in metrics.names.items()}
        rows.append(row)

    base = rows[0]
    for row in rows:
        row["delta_map50_95_pct"] = round(100 * (row["map50_95"] / base["map50_95"] - 1), 1)

    summary = ["file", "size_mb", "map50", "map50_95", "delta_map50_95_pct"]
    print("\n" + tabulate([{k: r[k] for k in summary} for r in rows], "keys", "github"))
    per_class = [k for k in rows[0] if k.startswith("map50_95_")]
    table = [[k.removeprefix("map50_95_")] + [r[k] for r in rows] for k in per_class]
    print("\nmAP50-95 per class\n")
    print(tabulate(table, ["class"] + [r["file"] for r in rows], "github"))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / ("evaluate.csv" if args.split == "val" else f"evaluate_{args.split}.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
