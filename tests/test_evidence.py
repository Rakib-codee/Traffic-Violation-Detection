import json

import cv2
import numpy as np
from scripts.make_sample_video import generate

from tvd.evidence import EventStore, EvidenceBuilder, blur_plate_band
from tvd.rules import ViolationEvent, ViolationType


def _event(start_frame=10, end_frame=12) -> ViolationEvent:
    return ViolationEvent(
        violation_type=ViolationType.RED_LIGHT,
        track_id=3,
        start_frame=start_frame,
        end_frame=end_frame,
        camera_id="test",
        details={"cross_point": (100.0, 100.0)},
    )


def test_blur_plate_band_changes_pixels():
    # checkerboard: blur visibly smooths it (a uniform fill would be a no-op)
    frame = np.indices((100, 100)).sum(axis=0) % 2 * 255
    frame = np.stack([frame] * 3, axis=-1).astype(np.uint8)
    out = blur_plate_band(frame, vehicle_boxes=[(20, 20, 80, 80)])
    # plate band (lower 20% of the box) is blurred, region above untouched
    assert not np.array_equal(out[68:80, 20:80], frame[68:80, 20:80])
    assert np.array_equal(out[20:50, 20:80], frame[20:50, 20:80])


def test_blur_disabled_returns_copy_unchanged():
    frame = np.full((50, 50, 3), 200, dtype=np.uint8)
    out = blur_plate_band(frame, vehicle_boxes=[])
    assert np.array_equal(out, frame)


def test_evidence_builder_extracts_clip(tmp_path):
    video = generate(tmp_path / "sample.mp4", seed=42)
    builder = EvidenceBuilder(video, tmp_path / "evidence", clip_seconds=4.0, blur=True)
    item = builder.build(_event(start_frame=20, end_frame=22))

    assert (tmp_path / "evidence" / item.event_id / "clip.mp4").exists()
    meta = json.loads((tmp_path / "evidence" / item.event_id / "metadata.json").read_text())
    assert meta["blurred"] is True
    assert "human review" in meta["note"]

    cap = cv2.VideoCapture(item.clip_path)
    frames = 0
    ok, _ = cap.read()
    while ok:
        frames += 1
        ok, _ = cap.read()
    cap.release()
    # 4s at 10fps -> 40 frames around the event (window 20-20..22+20)
    assert 30 <= frames <= 45


def test_event_store_crud(tmp_path):
    store = EventStore(tmp_path / "events.db")
    video = generate(tmp_path / "v.mp4", seed=42)
    item = EvidenceBuilder(video, tmp_path / "ev").build(_event())
    store.add_event(_event(), item)

    events = store.list_events()
    assert len(events) == 1
    assert events[0]["violation_type"] == "red_light"
    assert store.list_events(violation_type="wrong_way") == []
    assert store.get_event(item.event_id)["track_id"] == 3
    assert store.get_event("nope") is None
