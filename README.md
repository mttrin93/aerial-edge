# Aerial detection on the edge

Train YOLO26n on [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset), export to ONNX,
and compare **FP32 / FP16 / INT8** on accuracy (mAP) and latency with ONNX Runtime
(and TensorRT where a GPU is available).

## Pipeline

| Step | Where | Script | Output |
|------|-------|--------|--------|
| 2. Data and EDA | laptop | `scripts/eda.py` | box-size, class, objects-per-image plots |
| 3. Baseline training | Google Colab (GPU) | `scripts/train.py` | `best.pt`, val mAP |
| 4. ONNX export and parity check | laptop | `scripts/export.py` | `best.onnx`, max abs diff, val mAP |
| 5. Latency benchmark | laptop | `scripts/benchmark.py` | p50/p95 ms, size, peak memory |
| 6. Quantization (FP16, INT8 dynamic/static) | laptop | `scripts/quantize.py` | quantized `.onnx` variants |
| 6. mAP per variant | laptop | `scripts/evaluate.py` | val mAP50, mAP50-95 |
| 7. TensorRT (optional) | Colab GPU | `scripts/export.py` | `.engine` rows |

Settings live in `configs/`. All runs use `seed: 0`. Decisions are made on the val split;
test-dev is used once, at the end.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

The laptop install pulls CPU-only torch wheels (see `[tool.uv.sources]` in `pyproject.toml`).
On Colab, keep the preinstalled CUDA torch and install only the pinned Ultralytics:

```bash
pip install ultralytics==8.4.163
```

Get the data (~2 GB download, 3.7 GB on disk) and run the EDA:

```bash
.venv/bin/yolo settings datasets_dir=$PWD/datasets
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/eda.py
```

## Data: what makes VisDrone hard

VisDrone2019-DET: 6,471 train / 548 val / 1,610 test-dev images, 10 classes. The EDA
(`scripts/eda.py`, numbers in [`results/eda_summary.csv`](results/eda_summary.csv)) shows three
problems, and each one drives a later decision.

**1. Objects are tiny.** 60% of boxes are under 32 px (COCO "small") at the original resolution;
after the resize to the 640 px model input this becomes **91%**. Median box side at 640:
pedestrian 7.8 px, car 14 px, bus 21.6 px.
→ Resolution is the main accuracy lever (step 8), and aggressive quantization risks the
fine-grained features that tiny objects depend on.

![Box sizes](results/eda_box_sizes.png)

**2. Classes are imbalanced.** `car` is 42% of all boxes; `awning-tricycle` is 0.9% (45 to 1).
→ Report per-class mAP, not only the mean; rare classes are where INT8 degradation shows first.

![Class counts](results/eda_class_counts.png)

**3. Scenes are crowded.** 53 objects per train image and 71 per val image on average (busiest
image: 902), against about 7 in COCO.
→ YOLO26 is NMS-free, so crowding does not add NMS cost as in older YOLOs. The default cap of
`max_det=300` detections per image still truncates 3 of 548 val images (12 of 6,471 train);
keep the same `max_det` for every variant so mAP numbers stay comparable.

![Objects per image](results/eda_objects_per_image.png)

## Results

_TBD._

| Variant | Runtime | Size (MB) | mAP50 | mAP50-95 | p50 (ms) | p95 (ms) |
|---------|---------|-----------|-------|----------|----------|----------|

## Hardware

| Role | Device |
|------|--------|
| Training | _TBD (Colab GPU)_ |
| CPU benchmarks | Intel Core i7-8550U (4 cores / 8 threads, AVX2, no VNNI), Linux |

## License

Ultralytics is AGPL-3.0. That is fine for a public portfolio repo; a commercial product would
need an Ultralytics Enterprise license.
