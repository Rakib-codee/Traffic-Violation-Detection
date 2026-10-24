"""Helmet non-use rule (Phase 6 extension).

A secondary helmet detector runs on crops of motorcycle tracks. When a rider
is classified no-helmet for `min_frames` consecutive frames, a HELMET
violation event is raised once per track. The detector is injectable so the
rule is fully testable without model weights.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from tvd.config import HelmetConfig
from tvd.rules import ViolationEvent, ViolationType
from tvd.track.tracker import Track
from tvd.utils import get_logger

logger = get_logger(__name__)

# injected signature: (frame, crop_box) -> list[(label, confidence)]
CropDetector = Callable[[np.ndarray, tuple[int, int, int, int]], list[tuple[str, float]]]


class HelmetRule:
    """Stateful per-track helmet rule; feed motorcycle tracks per frame."""

    def __init__(self, config: HelmetConfig, detector: CropDetector, camera_id: str):
        if not config.enabled:
            raise ValueError("HelmetRule constructed while disabled; check config.enabled first")
        self.config = config
        self.detector = detector
        self.camera_id = camera_id
        self._runs: dict[int, int] = {}
        self._starts: dict[int, int] = {}
        self._flagged: set[int] = set()

    def update(self, frame: np.ndarray, frame_idx: int, track: Track) -> ViolationEvent | None:
        if track.track_id in self._flagged or track.class_name != "motorcycle":
            return None
        box = (int(track.x1), int(track.y1), int(track.x2), int(track.y2))
        detections = self.detector(frame, box)

        no_helmet = any(
            label.lower() in {lbl.lower() for lbl in self.config.no_helmet_labels}
            and conf >= self.config.conf_threshold
            for label, conf in detections
        )
        if no_helmet:
            if self._runs.get(track.track_id, 0) == 0:
                self._starts[track.track_id] = frame_idx
            self._runs[track.track_id] = self._runs.get(track.track_id, 0) + 1
        else:
            self._runs[track.track_id] = 0

        if self._runs.get(track.track_id, 0) >= self.config.min_frames:
            self._flagged.add(track.track_id)
            return ViolationEvent(
                violation_type=ViolationType.HELMET,
                track_id=track.track_id,
                start_frame=self._starts[track.track_id],
                end_frame=frame_idx,
                camera_id=self.camera_id,
                details={"evidence": "no-helmet detection on motorcycle rider crop"},
            )
        return None


def make_yolo_crop_detector(config: HelmetConfig, device: str = "auto") -> CropDetector:
    """Wrap a fine-tuned helmet model (YOLO) as a CropDetector."""
    from tvd.detect.detector import Detector, DetectorConfig, pick_device

    det_config = DetectorConfig(
        weights=config.weights, conf_threshold=config.conf_threshold, classes=[]
    )
    detector = Detector(det_config, device=pick_device(device))

    def detect(frame: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[str, float]]:
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return []
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (det_config.input_size, det_config.input_size))
        return [(d.class_name, d.confidence) for d in detector.predict(crop)]

    return detect
