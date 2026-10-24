"""Plate detection + OCR (ANPR). Injectable detector/OCR for testability."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tvd.config import AnprConfig
from tvd.utils import get_logger

logger = get_logger(__name__)

# (frame) -> list[(x1, y1, x2, y2, conf)]
PlateDetectorFn = Callable[[np.ndarray], list[tuple[int, int, int, int, float]]]
# (crop) -> list[(text, confidence)]
OcrFn = Callable[[np.ndarray], list[tuple[str, float]]]


@dataclass
class PlateRead:
    box: tuple[int, int, int, int]
    confidence: float
    text: str | None  # None unless AnprConfig.store_plate_text is true
    ocr_confidence: float | None
    stored: bool


class PlateReader:
    def __init__(self, config: AnprConfig, detector: PlateDetectorFn, ocr: OcrFn):
        if not config.enabled:
            raise ValueError("PlateReader constructed while disabled; check config.enabled first")
        self.config = config
        self.detector = detector
        self.ocr = ocr

    def read(self, frame: np.ndarray) -> list[PlateRead]:
        reads: list[PlateRead] = []
        for x1, y1, x2, y2, conf in self.detector(frame):
            if conf < self.config.conf_threshold:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ocr_results = self.ocr(crop)
            if not ocr_results:
                continue
            text, ocr_conf = max(ocr_results, key=lambda r: r[1])
            store = self.config.store_plate_text
            if not store:
                logger.info("plate read suppressed (store_plate_text=false)")
            reads.append(
                PlateRead(
                    box=(x1, y1, x2, y2),
                    confidence=conf,
                    text=text if store else None,
                    ocr_confidence=ocr_conf if store else None,
                    stored=store,
                )
            )
        return reads


def make_yolo_plate_detector(config: AnprConfig, device: str = "auto") -> PlateDetectorFn:
    from tvd.detect.detector import Detector, DetectorConfig, pick_device

    det_config = DetectorConfig(
        weights=config.weights, conf_threshold=config.conf_threshold, classes=[]
    )
    detector = Detector(det_config, device=pick_device(device))

    def detect(frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        return [
            (int(d.x1), int(d.y1), int(d.x2), int(d.y2), d.confidence)
            for d in detector.predict(frame)
        ]

    return detect


def make_easyocr_fn(config: AnprConfig) -> OcrFn:
    """easyocr downloads its model weights (~100MB) on first use."""
    import easyocr

    reader = easyocr.Reader(config.ocr_languages, gpu=False)

    def ocr(crop: np.ndarray) -> list[tuple[str, float]]:
        return [(text, conf) for _box, text, conf in reader.readtext(crop)]

    return ocr
