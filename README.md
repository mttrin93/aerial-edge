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
On Colab, keep the preinstalled CUDA torch and install only what training needs:

```bash
pip install ultralytics==8.4.163 mlflow==3.16.1
export PYTHONPATH=$PWD/src   # instead of `pip install -e .`, whose Python pin may not match Colab
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

## Baseline training (Colab)

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Colab (File > Open notebook >
GitHub, or upload it) and run the cells: it clones this repo, installs, downloads the data, and
runs the commands below. Set `REPO_URL` in cell 3 to your repo.

`scripts/train.py` passes `configs/train.yaml` to `YOLO.train()`; any `key=value` argument
overrides the config. On Colab, smoke test first, then the full run (do not train on the laptop:
even the smoke test freezes it):

```bash
python scripts/train.py epochs=1 fraction=0.01
```

On Colab the VM disk disappears on disconnect, so weights go to Google Drive and training can
resume from `last.pt`. MLflow uses a local SQLite DB (SQLite on a Drive mount is unreliable),
and `--backup-mlflow` copies it to Drive at the start of every epoch and once more at the end:

```bash
D=/content/drive/MyDrive/aerial-edge
export MLFLOW_TRACKING_URI=sqlite:////content/mlflow.db
python scripts/train.py project=$D/runs --backup-mlflow /content/mlflow.db $D/mlflow.db

# after a disconnect: restore the DB, then resume
cp $D/mlflow.db /content/mlflow.db
python scripts/train.py --resume $D/runs/yolo26n_640/weights/last.pt \
    --backup-mlflow /content/mlflow.db $D/mlflow.db
```

Without `MLFLOW_TRACKING_URI`, the script logs to `mlflow/mlflow.db` in the repo (git-ignored).
To browse the Colab runs on the laptop, copy `mlflow.db` from Drive into `mlflow/` and run
`mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`.

Note: at train time Ultralytics 8.4.163 raises `max_det` to the densest image in the data (902),
so its val mAP is not capped at 300 detections. Later steps must set `max_det` explicitly, to the
same value for every variant.

### Baseline result

YOLO26n, 640 px, 100 epochs, batch 16, on a Tesla T4: 4.6 h (about 2.5 min per epoch). The best
checkpoint (by fitness, 0.1·mAP50 + 0.9·mAP50-95) is from epoch 90.

| Split | P | R | mAP50 | mAP50-95 |
|-------|---|---|-------|----------|
| val (548 images, 38,759 boxes) | 0.462 | 0.354 | **0.342** | **0.187** |

Per class (val):

| Class | Boxes | mAP50 | mAP50-95 |
|-------|------:|------:|---------:|
| car | 14,064 | 0.751 | 0.495 |
| bus | 251 | 0.451 | 0.289 |
| motor | 4,886 | 0.405 | 0.164 |
| pedestrian | 8,844 | 0.374 | 0.155 |
| van | 1,975 | 0.368 | 0.242 |
| truck | 750 | 0.308 | 0.189 |
| people | 5,125 | 0.293 | 0.099 |
| tricycle | 1,045 | 0.232 | 0.124 |
| awning-tricycle | 532 | 0.129 | 0.076 |
| bicycle | 1,287 | 0.106 | 0.040 |

What the run shows:

- **In line with published nano baselines** on VisDrone at 640 (YOLOv8n / YOLO11n: mAP50 about
  0.33-0.35, mAP50-95 about 0.19-0.20).
- **Recall is the weak side** (0.35 against 0.46 precision): the model misses tiny objects more than
  it invents them, as the EDA predicts. The worst classes are the smallest or rarest (bicycle,
  awning-tricycle) and `people`, which is easy to confuse with `pedestrian`.
- **The curve flattens around epoch 60-70.** Val mAP50-95: 0.139 at epoch 10, 0.176 at 40, 0.183
  at 60, 0.187 at 90. About 60 epochs would give nearly the same model in half the time; more
  epochs will not help, resolution and model size are the levers left.
- **Turning mosaic off for the last 10 epochs (`close_mosaic=10`) did not help**: mAP50-95 dipped
  from 0.187 to 0.185-0.186 afterwards. On VisDrone mosaic probably keeps helping to the end, since
  it shows many small objects per batch; `close_mosaic=0` is worth a try in a later run.

## Results

_TBD._

| Variant | Runtime | Size (MB) | mAP50 | mAP50-95 | p50 (ms) | p95 (ms) |
|---------|---------|-----------|-------|----------|----------|----------|

## Hardware

| Role | Device |
|------|--------|
| Training | NVIDIA Tesla T4, 15 GB (Google Colab) |
| CPU benchmarks | Intel Core i7-8550U (4 cores / 8 threads, AVX2, no VNNI), Linux |

## License

Ultralytics is AGPL-3.0. That is fine for a public portfolio repo; a commercial product would
need an Ultralytics Enterprise license.
