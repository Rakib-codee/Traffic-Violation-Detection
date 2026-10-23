"""Config-driven object detector wrapper around ultralytics.

ultralytics is imported lazily so the rest of the library stays usable
without the heavy torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tvd.config import DetectorConfig
from tvd.utils import get_logger, set_seed

logger = get_logger(__name__)


@dataclass
class Detection:
    """One detected object in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


def pick_device(requested: str = "auto") -> str:
    """Resolve 'auto' to mps (Apple Silicon), cuda, or cpu."""
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Detector:
    """Runs a YOLO-family detector on frames and returns typed detections."""

    def __init__(self, config: DetectorConfig, device: str = "auto", seed: int = 42):
        self.config = config
        self.device = pick_device(device)
        set_seed(seed)
        from ultralytics import YOLO

        logger.info(
            "loading detector weights=%s device=%s conf=%.2f",
            config.weights,
            self.device,
            config.conf_threshold,
        )
        self._model = YOLO(str(config.weights))

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in one BGR frame."""
        results = self._model.predict(
            frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.input_size,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        names = results[0].names
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            class_name = names[class_id]
            if self.config.classes and class_name not in self.config.classes:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf.item()),
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return detections
