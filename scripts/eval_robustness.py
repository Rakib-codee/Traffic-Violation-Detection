"""Robustness evaluation: mAP on clean vs night/rain/glare validation images.

Builds augmented copies of the vehicle val split (labels unchanged, images
transformed deterministically with the system seed), evaluates the detector
on each, and writes docs/ROBUSTNESS.md + results/robustness/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from tvd.config import load_system_config, load_train_config
from tvd.detect.augment import TRANSFORMS
from tvd.detect.detector import pick_device
from tvd.utils import get_logger, set_seed

logger = get_logger(__name__)


def build_augmented_val(source_root: Path, out_root: Path, kind: str, seed: int) -> Path:
    """Copy the val split with transformed images; labels pass through."""
    images_src = source_root / "valid" / "images"
    labels_src = source_root / "valid" / "labels"
    out = Path(out_root) / kind
    images_dst, labels_dst = out / "valid" / "images", out / "valid" / "labels"
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(images_src.iterdir()):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        augmented = TRANSFORMS[kind](img, np.random.default_rng(seed + count))
        cv2.imwrite(str(images_dst / img_path.name), augmented)
        label = labels_src / f"{img_path.stem}.txt"
        if label.exists():
            shutil.copy(label, labels_dst / label.name)
        count += 1

    dataset_yaml = out / "data.yaml"
    dataset_yaml.write_text(
        f"path: {out.resolve()}\n"
        "train: valid/images\nval: valid/images\nnames: [bus, car, truck, van]\n"
    )
    logger.info("augmented val (%s): %d images", kind, count)
    return dataset_yaml


def evaluate(weights: str = "weights/detector.pt", device_override: str | None = None) -> dict:
    from ultralytics import YOLO

    system = load_system_config("configs/system.yaml")
    set_seed(system.seed)
    train_cfg = load_train_config("configs/train.yaml")
    device = pick_device(device_override or train_cfg.device)
    model = YOLO(weights)

    results: dict[str, dict[str, float]] = {}

    # clean baseline on the real val split
    clean = model.val(
        data="data/vehicles/data.yaml", device=device, seed=system.seed, verbose=False
    )
    results["clean"] = {"map50": float(clean.box.map50), "map": float(clean.box.map)}

    source_root = Path("data/vehicles")
    for kind in TRANSFORMS:
        dataset_yaml = build_augmented_val(source_root, Path("data/robustness"), kind, system.seed)
        metrics = model.val(data=str(dataset_yaml), device=device, seed=system.seed, verbose=False)
        results[kind] = {"map50": float(metrics.box.map50), "map": float(metrics.box.map)}

    out_dir = Path("results/robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps({"weights": weights, "seed": system.seed, "device": device, **results}, indent=2)
    )

    lines = [
        "# Robustness evaluation (Phase 6)",
        "",
        "Detector mAP on the UA-DETRAC-10K val split under deterministic degradations.",
        f"weights: `{weights}`  seed: {system.seed}  device: {device}",
        "",
        "| condition | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---|---|",
    ]
    for kind, r in results.items():
        lines.append(f"| {kind} | {r['map50']:.4f} | {r['map']:.4f} |")
    Path("docs/ROBUSTNESS.md").write_text("\n".join(lines) + "\n")
    logger.info("robustness: %s", {k: round(v["map50"], 3) for k, v in results.items()})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights/detector.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(args.weights, args.device)


if __name__ == "__main__":
    main()
