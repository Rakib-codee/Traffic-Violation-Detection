"""ANPR tests with injected detector/OCR (no model downloads in tests)."""

import numpy as np
import pytest

from tvd.config import AnprConfig
from tvd.plate import PlateReader

FRAME = np.zeros((200, 200, 3), dtype=np.uint8)


def fake_detector(box=(50, 50, 100, 70), conf=0.9):
    return lambda _frame: [(*box, conf)]


def fake_ocr(text="ABC123", conf=0.95):
    return lambda _crop: [(text, conf)]


def test_plate_text_not_stored_by_default():
    reader = PlateReader(AnprConfig(enabled=True), fake_detector(), fake_ocr())
    reads = reader.read(FRAME)
    assert len(reads) == 1
    assert reads[0].text is None
    assert reads[0].stored is False
    assert reads[0].confidence == 0.9


def test_plate_text_stored_when_enabled():
    reader = PlateReader(
        AnprConfig(enabled=True, store_plate_text=True), fake_detector(), fake_ocr("XYZ 99")
    )
    reads = reader.read(FRAME)
    assert reads[0].text == "XYZ 99"
    assert reads[0].stored is True


def test_low_confidence_detection_dropped():
    reader = PlateReader(
        AnprConfig(enabled=True, conf_threshold=0.5), fake_detector(conf=0.3), fake_ocr()
    )
    assert reader.read(FRAME) == []


def test_empty_ocr_no_read():
    reader = PlateReader(AnprConfig(enabled=True), fake_detector(), lambda _crop: [])
    assert reader.read(FRAME) == []


def test_disabled_config_rejected():
    with pytest.raises(ValueError, match="disabled"):
        PlateReader(AnprConfig(enabled=False), fake_detector(), fake_ocr())
