import csv

import cv2
import numpy as np

from tvd.config import TrackerConfig
from tvd.track.pipeline import track_video
from tvd.track.tracker import Track


class _FakeTracker:
    def __init__(self, tracks_by_frame):
        self._tracks_by_frame = tracks_by_frame

    def track_frame(self, frame, frame_idx):
        return self._tracks_by_frame.get(frame_idx, [])


def _write_video(path, n_frames=3):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(n_frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


def test_trajectories_csv_export(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_video(video)
    track = Track(
        track_id=1,
        x1=1.0,
        y1=2.0,
        x2=10.0,
        y2=20.0,
        confidence=0.9,
        class_name="car",
        frame_idx=1,
    )
    out_dir = tmp_path / "out"
    summary = track_video(
        video, _FakeTracker({1: [track]}), out_dir, TrackerConfig(), "fake.pt", annotate=False
    )

    assert summary["unique_track_ids"] == 1
    with (out_dir / "trajectories.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["frame"] == "1"
    assert rows[0]["track_id"] == "1"
    assert (out_dir / "summary.json").exists()
