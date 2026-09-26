"""Step 2: exploratory data analysis of the VisDrone labels.

Reads the YOLO-format labels of the train and val splits (test-dev stays untouched until the
end) and writes to results/:

  eda_box_sizes.png          object size, in original pixels and after resizing to the model input
  eda_class_counts.png       objects per class (class imbalance)
  eda_objects_per_image.png  how crowded the images are
  eda_summary.csv            per-class numbers behind the plots
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
from PIL import Image
from ultralytics.data.utils import check_det_dataset

from aerial_edge.paths import DATA_YAML, IMGSZ, RESULTS

SPLITS = ("train", "val")
SMALL, MEDIUM = 32, 96  # COCO size thresholds, on sqrt(box area) in pixels

# Palette (validated categorical slots 1-2, light surface) and text inks.
COLORS = {"train": "#2a78d6", "val": "#eb6834"}
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


@dataclass
class SplitStats:
    cls: np.ndarray  # class id per object
    side_px: np.ndarray  # sqrt(w*h) of each box in original pixels
    side_model: np.ndarray  # same box after letterbox resize to IMGSZ
    objects_per_image: np.ndarray


def load_split(root: Path, split: str) -> SplitStats:
    cls, side_px, side_model, counts = [], [], [], []
    for img_path in sorted((root / "images" / split).glob("*.jpg")):
        width, height = Image.open(img_path).size  # reads the header only
        label_path = root / "labels" / split / f"{img_path.stem}.txt"
        rows = np.loadtxt(label_path, ndmin=2) if label_path.stat().st_size else np.empty((0, 5))
        counts.append(len(rows))
        if not len(rows):
            continue
        side = np.sqrt(rows[:, 3] * width * rows[:, 4] * height)
        cls.append(rows[:, 0].astype(int))
        side_px.append(side)
        side_model.append(side * IMGSZ / max(width, height))
    return SplitStats(
        np.concatenate(cls), np.concatenate(side_px), np.concatenate(side_model), np.array(counts)
    )


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(width: float = 8, height: float = 4.5):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    style_axes(ax)
    return fig, ax


def title(ax: plt.Axes, headline: str, subtitle: str) -> None:
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK_MUTED, pad=10)
    ax.text(0, 1.1, headline, transform=ax.transAxes, fontsize=12.5, color=INK, weight="bold")


def plot_box_sizes(stats: SplitStats, out: Path) -> None:
    fig, ax = new_figure()
    bins = np.logspace(0, np.log10(1000), 60)
    series = {
        "original resolution": (stats.side_px, INK_MUTED),
        f"resized to {IMGSZ} (model input)": (stats.side_model, COLORS["train"]),
    }
    for label, (values, color) in series.items():
        hist, edges = np.histogram(values, bins=bins)
        ax.stairs(hist / len(values) * 100, edges, color=color, linewidth=2, label=label)
    for threshold, name in ((SMALL, "small < 32 px"), (MEDIUM, "medium < 96 px")):
        ax.axvline(threshold, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(threshold * 1.06, ax.get_ylim()[1] * 0.95, name, color=INK_MUTED, fontsize=8.5,
                va="top")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("box size, √(width × height) in pixels (log scale)", color=INK_MUTED)
    ax.set_ylabel("% of objects", color=INK_MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    small_orig = (stats.side_px < SMALL).mean() * 100
    small_model = (stats.side_model < SMALL).mean() * 100
    title(ax, f"{small_model:.0f}% of objects are smaller than 32 px at the model input",
          f"Train split, {len(stats.side_px):,} boxes. At original resolution: {small_orig:.0f}%.")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_class_counts(stats: SplitStats, names: dict[int, str], out: Path) -> None:
    counts = np.bincount(stats.cls, minlength=len(names))
    order = np.argsort(counts)  # ascending, so the biggest bar ends up on top
    fig, ax = new_figure(height=4.8)
    ax.barh([names[i] for i in order], counts[order], color=COLORS["train"], height=0.7)
    ax.grid(axis="y", visible=False)
    for y, i in enumerate(order):
        share = counts[i] / counts.sum() * 100
        ax.text(counts[i], y, f"  {counts[i]:,} ({share:.1f}%)", va="center", fontsize=8.5,
                color=INK_MUTED)
    ax.set_xlim(0, counts.max() * 1.22)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("objects in train split", color=INK_MUTED)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=9.5)
    most, least = order[-1], order[0]
    title(ax, f"{names[most]} outnumbers {names[least]} {counts[most] / counts[least]:.0f} to 1",
          "Class imbalance in the train split")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_objects_per_image(stats: dict[str, SplitStats], out: Path) -> None:
    fig, ax = new_figure()
    top = max(s.objects_per_image.max() for s in stats.values())
    bins = np.arange(0, top + 20, 20)
    x_max = 400  # a handful of very crowded images would otherwise stretch the axis
    for split, s in stats.items():
        hist, edges = np.histogram(s.objects_per_image, bins=bins)
        mean = s.objects_per_image.mean()
        ax.stairs(hist / len(s.objects_per_image) * 100, edges, color=COLORS[split], linewidth=2,
                  label=f"{split} ({len(s.objects_per_image):,} images, mean {mean:.0f})")
        ax.axvline(mean, color=COLORS[split], linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xlim(0, x_max)
    ax.set_xlabel("objects per image", color=INK_MUTED)
    ax.set_ylabel("% of images", color=INK_MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    means = {k: s.objects_per_image.mean() for k, s in stats.items()}
    title(ax, f"A val image holds {means['val']:.0f} objects on average",
          f"Dashed lines mark the mean; axis cut at {x_max}, busiest image has {top}. "
          "COCO averages about 7.")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_summary(stats: dict[str, SplitStats], names: dict[int, str], out: Path) -> None:
    train, val = stats["train"], stats["val"]
    train_counts = np.bincount(train.cls, minlength=len(names))
    val_counts = np.bincount(val.cls, minlength=len(names))
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "train_objects", "train_share_pct", "val_objects",
                         f"median_side_px_at_{IMGSZ}", f"pct_small_at_{IMGSZ}"])
        for i, name in names.items():
            sides = train.side_model[train.cls == i]
            writer.writerow([name, train_counts[i], round(train_counts[i] / len(train.cls) * 100, 2),
                             val_counts[i], round(float(np.median(sides)), 1),
                             round(float((sides < SMALL).mean() * 100), 1)])


def main() -> None:
    info = check_det_dataset(DATA_YAML)
    root, names = Path(info["path"]), info["names"]
    RESULTS.mkdir(exist_ok=True)

    stats = {split: load_split(root, split) for split in SPLITS}
    for split, s in stats.items():
        unknown = np.setdiff1d(np.unique(s.cls), list(names))
        assert not len(unknown), f"{split}: class ids {unknown} not in {DATA_YAML}"
        empty = (s.objects_per_image == 0).sum()
        print(f"{split}: {len(s.objects_per_image)} images ({empty} without labels), "
              f"{len(s.cls)} objects, {s.objects_per_image.mean():.1f} per image, "
              f"{(s.side_model < SMALL).mean():.0%} small at {IMGSZ}")

    plot_box_sizes(stats["train"], RESULTS / "eda_box_sizes.png")
    plot_class_counts(stats["train"], names, RESULTS / "eda_class_counts.png")
    plot_objects_per_image(stats, RESULTS / "eda_objects_per_image.png")
    write_summary(stats, names, RESULTS / "eda_summary.csv")
    print(f"Wrote plots and eda_summary.csv to {RESULTS}")


if __name__ == "__main__":
    main()
