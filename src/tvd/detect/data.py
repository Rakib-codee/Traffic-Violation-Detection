"""YOLO-format dataset validation and preparation helpers.

Datasets live in the gitignored data/ dir in ultralytics YOLO layout:

    <root>/data.yaml          # path, train, val, names
    <root>/images/train/*.jpg
    <root>/labels/train/*.txt  # one line per object: <class_id> <cx> <cy> <w> <h> (normalized)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from tvd.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetInfo:
    root: Path
    names: list[str]
    n_train: int
    n_val: int


def load_dataset_yaml(dataset_yaml: Path) -> dict:
    with Path(dataset_yaml).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for key in ("path", "train", "val", "names"):
        if key not in data:
            raise ValueError(f"{dataset_yaml}: missing required key '{key}'")
    return data


def validate_label_file(path: Path, n_classes: int) -> int:
    """Check one YOLO label file; returns the number of objects."""
    count = 0
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{lineno}: expected 5 fields, got {len(parts)}")
        class_id = int(parts[0])
        if not 0 <= class_id < n_classes:
            raise ValueError(
                f"{path}:{lineno}: class id {class_id} out of range 0..{n_classes - 1}"
            )
        coords = [float(v) for v in parts[1:]]
        if any(not 0.0 <= v <= 1.0 for v in coords):
            raise ValueError(f"{path}:{lineno}: coordinates must be normalized to [0, 1]")
        if coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"{path}:{lineno}: box width/height must be positive")
        count += 1
    return count


def _count_images(images_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(1 for p in images_dir.iterdir() if p.suffix.lower() in exts)


def validate_dataset(dataset_yaml: Path) -> DatasetInfo:
    """Validate a YOLO dataset end to end; raises on the first problem found."""
    dataset_yaml = Path(dataset_yaml)
    data = load_dataset_yaml(dataset_yaml)
    root = Path(data["path"])
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    names = list(data["names"].values()) if isinstance(data["names"], dict) else list(data["names"])

    counts: dict[str, int] = {}
    for split in ("train", "val"):
        default_dir = root / "images" / split
        images_dir = default_dir if default_dir.exists() else root / data[split]
        labels_dir = root / "labels" / split
        if not images_dir.is_dir():
            raise ValueError(
                f"{dataset_yaml}: missing images dir for split '{split}': {images_dir}"
            )
        if not labels_dir.is_dir():
            raise ValueError(
                f"{dataset_yaml}: missing labels dir for split '{split}': {labels_dir}"
            )
        n_objects = 0
        for label_file in sorted(labels_dir.glob("*.txt")):
            n_objects += validate_label_file(label_file, len(names))
        counts[split] = _count_images(images_dir)
        logger.info(
            "split=%s images=%d label_files=%d objects=%d",
            split,
            counts[split],
            len(list(labels_dir.glob("*.txt"))),
            n_objects,
        )

    return DatasetInfo(root=root, names=names, n_train=counts["train"], n_val=counts["val"])
