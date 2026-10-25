"""Run detector+tracker over a video and export tracks/annotated video."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from tvd.config import TrackerConfig
from tvd.track.mot import track_to_row, write_mot
from tvd.track.tracker import Track, Tracker
from tvd.utils import get_logger

logger = get_logger(__name__)


def _annotate(frame, tracks: list[Track]) -> None:
    for t in tracks:
        x1, y1, x2, y2 = int(t.x1), int(t.y1), int(t.x2), int(t.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            frame,
            f"{t.track_id}:{t.class_name}",
            (x1, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
        )


def track_video(
    video_path: Path,
    tracker: Tracker,
    out_dir: Path,
    tracker_config: TrackerConfig,
    weights: str,
    annotate: bool = True,
) -> dict:
    """Track every frame of a video; write MOT tracks, trajectories CSV,
    annotated video (optional), and a summary. Returns the summary dict."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if annotate:
        writer = cv2.VideoWriter(
            str(out_dir / "annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    all_tracks: list[Track] = []
    n_frames = 0
    t0 = time.perf_counter()
    ok, frame = cap.read()
    while ok:
        n_frames += 1
        tracks = tracker.track_frame(frame, n_frames)
        all_tracks.extend(tracks)
        if writer is not None:
            _annotate(frame, tracks)
            writer.write(frame)
        ok, frame = cap.read()
    elapsed = time.perf_counter() - t0

    cap.release()
    if writer is not None:
        writer.release()

    rows = [track_to_row(t) for t in all_tracks]
    write_mot(rows, out_dir / "tracks.txt")

    with (out_dir / "trajectories.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["frame", "track_id", "x1", "y1", "x2", "y2", "confidence", "class_name"]
        )
        w.writeheader()
        for t in all_tracks:
            row = asdict(t)
            row["frame"] = row.pop("frame_idx")
            w.writerow(row)

    long_tracks = {t.track_id for t in all_tracks}
    summary = {
        "video": str(video_path),
        "weights": weights,
        "algorithm": tracker_config.algorithm,
        "frames": n_frames,
        "unique_track_ids": len(long_tracks),
        "detections": len(all_tracks),
        "elapsed_s": round(elapsed, 2),
        "fps": round(n_frames / elapsed, 2) if elapsed > 0 else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "tracked %d frames, %d unique ids, %.1f fps", n_frames, len(long_tracks), summary["fps"]
    )
    return summary
