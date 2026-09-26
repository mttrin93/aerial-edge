"""Step 6: quantize the FP32 ONNX model into FP16, INT8 dynamic and INT8 static variants.

    python scripts/quantize.py                     # all three
    python scripts/quantize.py int8_static         # just one

Starts from models/best.onnx (step 4) and writes, next to it:
  best_fp16.onnx          weights and activations in float16, float32 input/output
  best_int8_dynamic.onnx  int8 weights; activation ranges computed at run time, per image
  best_int8_static.onnx   int8 weights and activations; activation ranges fixed beforehand from
                          calibration images (QDQ format, which ORT fuses into int8 kernels)

Calibration uses a seeded random sample of *train* images, never val or test, so the ranges do
not leak the evaluation data. Settings in the `calibration` and `quantize` sections of
configs/export.yaml. Accuracy: scripts/evaluate.py; speed: scripts/benchmark.py.
"""

import argparse
import random
import re
import tempfile
from pathlib import Path

import cv2
import onnx
import yaml
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from aerial_edge.inference import preprocess, split_images
from aerial_edge.paths import CONFIGS, MODELS

VARIANTS = ("fp16", "int8_dynamic", "int8_static")


class ImageReader(CalibrationDataReader):
    """Feeds calibration images one at a time, so only one is in memory."""

    def __init__(self, paths: list[Path], input_name: str, imgsz: int):
        self.paths = iter(paths)
        self.input_name = input_name
        self.imgsz = imgsz

    def get_next(self) -> dict | None:
        path = next(self.paths, None)
        if path is None:
            return None
        return {self.input_name: preprocess(cv2.imread(str(path)), self.imgsz)}


def copy_metadata(src: onnx.ModelProto, dst_path: Path) -> None:
    """Ultralytics reads class names, stride and imgsz from the ONNX metadata; the quantizers
    drop it, so copy it over."""
    dst = onnx.load(dst_path)
    del dst.metadata_props[:]
    dst.metadata_props.extend(src.metadata_props)
    onnx.save(dst, dst_path)


def to_fp16(src: Path, dst: Path) -> None:
    model = onnx.load(src)
    # The export's stored float32 type info contradicts the Casts the converter inserts (ORT then
    # refuses to load the model at a Resize node); drop it and let ORT infer types again.
    del model.graph.value_info[:]
    # keep_io_types: the model still takes and returns float32, so callers need no change
    onnx.save(float16.convert_float_to_float16(model, keep_io_types=True), dst)


def excluded_nodes(model_path: Path, patterns: list[str]) -> list[str]:
    """Names of the nodes that match any of the regex patterns; they stay in float32."""
    names = [n.name for n in onnx.load(model_path).graph.node]
    return [n for n in names if any(re.fullmatch(p, n) for p in patterns)]


def to_int8_dynamic(src: Path, dst: Path, q: dict) -> None:
    # reduce_range and exclude are tuned for static INT8; dynamic INT8 got worse with
    # reduce_range (results/quantization_experiments.csv), so it keeps full 8-bit weights.
    quantize_dynamic(
        src,
        dst,
        op_types_to_quantize=q["op_types"],
        per_channel=q["per_channel"],
        reduce_range=False,
        weight_type=QuantType[q["weight_type"]],
    )


def to_int8_static(src: Path, dst: Path, cfg: dict) -> None:
    q, calib = cfg["quantize"], cfg["calibration"]
    paths = split_images(calib["split"])
    paths = random.Random(cfg["seed"]).sample(paths, calib["num_images"])
    input_name = onnx.load(src).graph.input[0].name
    quantize_static(
        src,
        dst,
        ImageReader(paths, input_name, cfg["imgsz"]),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=q["op_types"],
        nodes_to_exclude=excluded_nodes(src, q["exclude"]),
        per_channel=q["per_channel"],
        reduce_range=q["reduce_range"],
        activation_type=QuantType[q["activation_type"]],
        weight_type=QuantType[q["weight_type"]],
        # MinMax keeps only per-tensor min/max per image, so 300 images fit in memory. Do not set
        # CalibMaxIntermediateOutputs: in ORT 1.30 it discards the collected data at the cap
        # instead of folding it into the ranges.
        calibrate_method=CalibrationMethod[calib["method"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("variants", nargs="*", help=f"any of {', '.join(VARIANTS)} (default: all)")
    parser.add_argument("--model", type=Path, default=MODELS / "best.onnx")
    parser.add_argument("--config", type=Path, default=CONFIGS / "export.yaml")
    parser.add_argument("--tag", default="", help="appended to output names, for experiments")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    variants = args.variants or VARIANTS
    if unknown := set(variants) - set(VARIANTS):
        parser.error(f"unknown variant(s): {', '.join(sorted(unknown))}")
    source = onnx.load(args.model)

    with tempfile.TemporaryDirectory() as tmp:
        # ORT's recommended preparation for INT8: shape inference and graph optimisation, so the
        # quantizer sees fused ops and every tensor's shape.
        prepared = Path(tmp) / "prepared.onnx"
        quant_pre_process(args.model, prepared)

        for variant in variants:
            dst = args.model.with_name(f"{args.model.stem}_{variant}{args.tag}.onnx")
            print(f"{variant}: writing {dst.name} ...", flush=True)
            if variant == "fp16":
                to_fp16(args.model, dst)
            elif variant == "int8_dynamic":
                to_int8_dynamic(prepared, dst, cfg["quantize"])
            else:
                to_int8_static(prepared, dst, cfg)
            copy_metadata(source, dst)
            print(f"  {dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
