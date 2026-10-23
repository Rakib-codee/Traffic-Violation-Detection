"""Fine-tune the vehicle detector.

Validates the dataset, trains with the seeded config, and copies the best
weights to weights/detector.pt. Every run logs its full config and seed to
results/detection/<run_name>/run_config.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tvd.config import load_system_config, load_train_config
from tvd.detect.data import validate_dataset
from tvd.detect.detector import pick_device
from tvd.utils import get_logger, set_seed

logger = get_logger(__name__)


def train(
    train_config_path: str = "configs/train.yaml",
    system_config_path: str = "configs/system.yaml",
    epochs_override: int | None = None,
    device_override: str | None = None,
) -> Path:
    from ultralytics import YOLO

    system = load_system_config(system_config_path)
    cfg = load_train_config(train_config_path)
    set_seed(system.seed)

    info = validate_dataset(cfg.dataset_yaml)
    logger.info(
        "dataset: %d train / %d val images, classes=%s", info.n_train, info.n_val, info.names
    )

    device = pick_device(device_override or cfg.device)
    # ultralytics prefixes relative project paths with runs/detect; pass absolute.
    run_dir = (system.results_dir / "detection" / cfg.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "seed": system.seed,
        "device": device,
        **cfg.model_dump(mode="json"),
        "epochs": epochs_override or cfg.epochs,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    logger.info("run config: %s", run_config)

    model = YOLO(str(cfg.base_weights))
    results = model.train(
        data=str(cfg.dataset_yaml),
        epochs=epochs_override or cfg.epochs,
        batch=cfg.batch_size,
        imgsz=cfg.input_size,
        optimizer=cfg.optimizer,
        lr0=cfg.learning_rate,
        device=device,
        seed=system.seed,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        verbose=False,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training finished but best weights not found at {best}")
    dest = Path("weights/detector.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, dest)
    logger.info("best weights copied to %s", dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--system-config", default="configs/system.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs (smoke runs)")
    parser.add_argument("--device", default=None, help="override device (mps/cuda/cpu)")
    args = parser.parse_args()
    train(args.train_config, args.system_config, args.epochs, args.device)


if __name__ == "__main__":
    main()
