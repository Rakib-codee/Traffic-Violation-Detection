"""Helmet rule tests with a synthetic crop detector (no model weights)."""

import numpy as np
import pytest

from tvd.config import HelmetConfig
from tvd.rules import ViolationType
from tvd.rules.helmet import HelmetRule
from tvd.track.tracker import Track

FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


def _moto(track_id=1, frame_idx=1) -> Track:
    return Track(
        track_id=track_id,
        x1=10,
        y1=10,
        x2=50,
        y2=80,
        confidence=0.9,
        class_name="motorcycle",
        frame_idx=frame_idx,
    )


def _car(track_id=2, frame_idx=1) -> Track:
    t = _moto(track_id, frame_idx)
    t.class_name = "car"
    return t


def fake_detector(labels: list[str]):
    def detect(_frame, _box):
        return [(label, 0.9) for label in labels]

    return detect


def _config(min_frames=5) -> HelmetConfig:
    return HelmetConfig(enabled=True, min_frames=min_frames, conf_threshold=0.25)


def test_no_helmet_flags_after_min_frames():
    rule = HelmetRule(_config(min_frames=3), fake_detector(["without helmet"]), "cam")
    events = [rule.update(FRAME, f, _moto(frame_idx=f)) for f in range(1, 6)]
    fired = [e for e in events if e]
    assert len(fired) == 1
    assert fired[0].violation_type is ViolationType.HELMET
    assert fired[0].end_frame == 3


def test_helmet_present_no_event():
    rule = HelmetRule(_config(), fake_detector(["with helmet"]), "cam")
    for f in range(1, 10):
        assert rule.update(FRAME, f, _moto(frame_idx=f)) is None


def test_interrupted_run_resets():
    rule = HelmetRule(_config(min_frames=3), fake_detector([]), "cam")

    def detector_with_gap(_frame, box):
        return [("without helmet", 0.9)]

    # two no-helmet frames, one clean frame, two more -> never reaches 3
    rule.detector = fake_detector(["without helmet"])
    assert rule.update(FRAME, 1, _moto(frame_idx=1)) is None
    assert rule.update(FRAME, 2, _moto(frame_idx=2)) is None
    rule.detector = fake_detector(["with helmet"])
    assert rule.update(FRAME, 3, _moto(frame_idx=3)) is None
    rule.detector = fake_detector(["without helmet"])
    assert rule.update(FRAME, 4, _moto(frame_idx=4)) is None
    assert rule.update(FRAME, 5, _moto(frame_idx=5)) is None  # run of 2 only


def test_car_tracks_ignored():
    rule = HelmetRule(_config(min_frames=1), fake_detector(["without helmet"]), "cam")
    assert rule.update(FRAME, 1, _car(frame_idx=1)) is None


def test_disabled_config_rejected():
    with pytest.raises(ValueError, match="disabled"):
        HelmetRule(HelmetConfig(enabled=False), fake_detector([]), "cam")


def test_flags_only_once_per_track():
    rule = HelmetRule(_config(min_frames=2), fake_detector(["without helmet"]), "cam")
    assert rule.update(FRAME, 1, _moto(frame_idx=1)) is None
    assert rule.update(FRAME, 2, _moto(frame_idx=2)) is not None
    assert rule.update(FRAME, 3, _moto(frame_idx=3)) is None
