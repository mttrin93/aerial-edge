"""Download VisDrone2019-DET and convert its labels to YOLO format (~2 GB).

Uses the download script embedded in Ultralytics' VisDrone.yaml. The data lands in the
Ultralytics `datasets_dir` setting; for this repo set it once with:

    yolo settings datasets_dir=$PWD/datasets
"""

from ultralytics.data.utils import check_det_dataset

from aerial_edge.paths import DATA_YAML


def main() -> None:
    info = check_det_dataset(DATA_YAML, autodownload=True)
    print(f"Dataset ready at {info['path']}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {info.get(split)}")


if __name__ == "__main__":
    main()
