"""Evaluate the detector and write the mAP table.

Outputs (regenerated from real runs only, both committed):
  results/detection/<run_name>/metrics.json
  results/detection/<run_name>/mAP.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tvd.config import load_detector_config, load_system_config, load_train_config
from tvd.detect.data import validate_dataset
from tvd.detect.detector import pick_device
from tvd.utils import get_logger, set_seed

logger = get_logger(__name__)


def _format_table(names: list[str], per_class: dict[str, dict[str, float]]) -> str:
    lines = [
        "| class | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---|---|",
    ]
    for name in names:
        row = per_class.get(name)
        if row:
            lines.append(f"| {name} | {row['map50']:.4f} | {row['map']:.4f} |")
    overall = per_class.get("all")
    if overall:
        lines.append(f"| **all** | **{overall['map50']:.4f}** | **{overall['map']:.4f}** |")
    return "\n".join(lines)


def evaluate(
    weights: str = "weights/detector.pt",
    train_config_path: str = "configs/train.yaml",
    system_config_path: str = "configs/system.yaml",
    device_override: str | None = None,
) -> dict:
    from ultralytics import YOLO

    system = load_system_config(system_config_path)
    train_cfg = load_train_config(train_config_path)
    det_cfg = load_detector_config("configs/model.yaml")
    set_seed(system.seed)

    info = validate_dataset(train_cfg.dataset_yaml)
    device = pick_device(device_override or train_cfg.device)

    model = YOLO(weights)
    out_dir = (system.results_dir / "detection" / train_cfg.run_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = model.val(
        data=str(train_cfg.dataset_yaml),
        imgsz=det_cfg.input_size,
        device=device,
        seed=system.seed,
        project=str(out_dir),
        name="val",
        exist_ok=True,
        verbose=False,
    )

    # only classes present in the val labels get metric rows
    present = [int(i) for i in metrics.box.ap_class_index]
    names = [metrics.names[i] for i in present]
    per_class: dict[str, dict[str, float]] = {
        "all": {"map50": float(metrics.box.map50), "map": float(metrics.box.map)}
    }
    for i, class_id in enumerate(present):
        per_class[metrics.names[class_id]] = {
            "map50": float(metrics.box.ap50[i]),
            "map": float(metrics.box.maps[i]),
        }
    payload = {
        "weights": weights,
        "dataset": str(train_cfg.dataset_yaml),
        "seed": system.seed,
        "device": device,
        "n_val_images": info.n_val,
        "classes": names,
        "per_class": per_class,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    table = _format_table(names, per_class)
    report = (
        f"# Detector mAP\n\nrun: `{train_cfg.run_name}`\nweights: `{weights}`\n"
        f"dataset: `{train_cfg.dataset_yaml}`\nseed: {system.seed}\ndevice: {device}\n"
        f"val images: {info.n_val}\n\n{table}\n"
    )
    (out_dir / "mAP.md").write_text(report)
    # results/ is gitignored; the committed headline table lives in docs/.
    docs_results = Path("docs/RESULTS.md")
    header = "# Results\n\nRegenerated from real runs; do not edit by hand.\n\n"
    docs_results.write_text(header + report)
    logger.info(
        "mAP@0.5=%.4f mAP@0.5:0.95=%.4f", per_class["all"]["map50"], per_class["all"]["map"]
    )
    print(table)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights/detector.pt")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--system-config", default="configs/system.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(args.weights, args.train_config, args.system_config, args.device)


if __name__ == "__main__":
    main()
