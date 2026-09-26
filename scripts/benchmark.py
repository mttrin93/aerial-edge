"""Step 5: CPU latency and memory of each model variant, preprocessing to final boxes.

    python scripts/benchmark.py                                  # best.pt and best.onnx
    python scripts/benchmark.py models/best.onnx models/best_fp16.onnx

For every image the pipeline is timed in three stages, then as a whole:
  pre    letterbox the raw val image to 640x640, BGR->RGB, CHW, float [0, 1]
  infer  the model itself (ONNX Runtime session.run, or the PyTorch forward pass)
  nms    Ultralytics non_max_suppression on the raw output, at the deployment conf threshold
The exported ONNX graph ends before NMS, so timing only `infer` would understate the real cost.

Each model runs in its own process, so peak memory is not mixed between models. Peak memory is
the growth of the process's peak RSS from just before loading the model to the end of the runs.
Settings (threads, warmup, runs, conf, iou) are in the `benchmark` section of
configs/export.yaml. Writes results/benchmark.csv.
"""

import argparse
import csv
import multiprocessing as mp
import platform
import resource
import time
from pathlib import Path
from queue import Empty

import numpy as np
import yaml
from tabulate import tabulate

from aerial_edge.inference import preprocess, split_images
from aerial_edge.paths import CONFIGS, MODELS, RESULTS


def describe(path: Path) -> tuple[str, str]:
    """(variant, runtime) from the file name: best.onnx -> FP32, best_int8_static.onnx -> INT8..."""
    runtime = "PyTorch" if path.suffix == ".pt" else "ONNX Runtime"
    _, _, tag = path.stem.partition("_")
    return (tag.replace("_", " ").upper() or "FP32"), runtime


def load_images(n: int) -> list[np.ndarray]:
    import cv2

    return [cv2.imread(str(p)) for p in split_images("val")[:n]]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KiB


def run_one(path: Path, cfg: dict, queue: mp.Queue) -> None:
    """Benchmark one model; runs in a child process and puts its stats on the queue."""
    import torch
    from ultralytics.utils.nms import non_max_suppression

    bench = cfg["benchmark"]
    threads = bench["intra_op_threads"]
    torch.set_num_threads(threads)
    images = load_images(bench["images"])

    rss_before = peak_rss_mb()
    if path.suffix == ".pt":
        from ultralytics import YOLO

        model = YOLO(path).model.fuse().eval().float()

        def infer(x: np.ndarray) -> torch.Tensor:
            with torch.inference_mode():
                y = model(torch.from_numpy(x))
            return y[0] if isinstance(y, (list, tuple)) else y
    else:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        def infer(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(session.run(None, {input_name: x})[0])

    def nms(y: torch.Tensor) -> list[torch.Tensor]:
        return non_max_suppression(
            y.float(), conf_thres=bench["conf"], iou_thres=bench["iou"], max_det=cfg["max_det"]
        )

    times = {"pre": [], "infer": [], "nms": [], "total": []}
    boxes = []
    for i in range(bench["warmup"] + bench["runs"]):
        image = images[i % len(images)]
        t0 = time.perf_counter()
        x = preprocess(image, cfg["imgsz"])
        t1 = time.perf_counter()
        y = infer(x)
        t2 = time.perf_counter()
        detections = nms(y)
        t3 = time.perf_counter()
        if i >= bench["warmup"]:
            for key, dt in zip(times, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)):
                times[key].append(dt * 1000)
            boxes.append(len(detections[0]))

    queue.put(
        {
            "times": {k: np.array(v) for k, v in times.items()},
            "boxes_per_image": float(np.mean(boxes)),
            "peak_mem_mb": peak_rss_mb() - rss_before,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "models", nargs="*", type=Path, default=[MODELS / "best.pt", MODELS / "best.onnx"]
    )
    parser.add_argument("--config", type=Path, default=CONFIGS / "export.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    bench = cfg["benchmark"]
    assert bench["batch"] == 1, "only batch 1 is implemented"

    print(
        f"{platform.processor() or platform.machine()}, {bench['intra_op_threads']} threads, "
        f"{bench['warmup']} warmup + {bench['runs']} runs over {bench['images']} val images, "
        f"NMS conf={bench['conf']} iou={bench['iou']}\n"
    )
    ctx = mp.get_context("spawn")
    rows = []
    for path in args.models:
        queue = ctx.Queue()
        process = ctx.Process(target=run_one, args=(path, cfg, queue))
        process.start()
        while True:  # a crashed child never puts anything on the queue: do not wait forever
            try:
                stats = queue.get(timeout=5)
                break
            except Empty:
                if not process.is_alive():
                    raise RuntimeError(f"{path.name}: benchmark process died") from None
        process.join()

        t = stats["times"]
        variant, runtime = describe(path)
        rows.append(
            {
                "variant": variant,
                "runtime": runtime,
                "file": path.name,
                "size_mb": round(path.stat().st_size / 1e6, 2),
                "pre_p50_ms": round(float(np.percentile(t["pre"], 50)), 2),
                "infer_p50_ms": round(float(np.percentile(t["infer"], 50)), 2),
                "infer_p95_ms": round(float(np.percentile(t["infer"], 95)), 2),
                "nms_p50_ms": round(float(np.percentile(t["nms"], 50)), 2),
                "total_p50_ms": round(float(np.percentile(t["total"], 50)), 2),
                "total_p95_ms": round(float(np.percentile(t["total"], 95)), 2),
                "boxes_per_image": round(stats["boxes_per_image"], 1),
                "peak_mem_mb": round(stats["peak_mem_mb"], 1),
                "threads": bench["intra_op_threads"],
            }
        )
        print(f"done: {path.name}")

    print("\n" + tabulate(rows, headers="keys", tablefmt="github"))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "benchmark.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
