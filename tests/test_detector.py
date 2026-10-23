import cv2
import numpy as np
import pytest

from tvd.config import DetectorConfig
from tvd.detect.detector import Detection, Detector, pick_device


def _bus_frame():
    from ultralytics import ASSETS

    frame = cv2.imread(str(ASSETS / "bus.jpg"))
    assert frame is not None
    return frame


@pytest.fixture(scope="module")
def detector():
    # yolov8n.pt auto-downloads (~6 MB) on first use
    cfg = DetectorConfig(weights="yolov8n.pt", conf_threshold=0.25)
    return Detector(cfg, device="auto", seed=42)


def test_pick_device_auto():
    assert pick_device("auto") in {"mps", "cuda", "cpu"}
    assert pick_device("cpu") == "cpu"


def test_detector_finds_bus(detector):
    detections = detector.predict(_bus_frame())
    assert all(isinstance(d, Detection) for d in detections)
    names = {d.class_name for d in detections}
    assert "bus" in names
    for d in detections:
        assert d.x2 > d.x1 and d.y2 > d.y1
        assert 0.0 < d.confidence <= 1.0


def test_detector_class_filter():
    cfg = DetectorConfig(weights="yolov8n.pt", classes=["car"], conf_threshold=0.1)
    det = Detector(cfg, device="auto", seed=42)
    for d in det.predict(_bus_frame()):
        assert d.class_name == "car"


def test_detector_blank_frame_returns_list(detector):
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    assert detector.predict(blank) == []
