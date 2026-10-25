"""Shared core pipeline: video -> track -> signal -> rules -> events.

Used by scripts/eval_events.py (event-level eval) and scripts/eval_dropout.py
(robustness study). scripts/run_pipeline.py has its own loop because it also
builds evidence and writes the event DB.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from tvd.config import EventClip, load_scene_config, load_tracker_config
from tvd.rules import RuleEngine, ViolationEvent
from tvd.signal import SignalSmoother
from tvd.track.tracker import Tracker
from tvd.utils import get_logger

logger = get_logger(__name__)


def run_clip(
    clip: EventClip,
    tracker_config_path: str,
    device: str,
    work_dir: Path,
    stitch: bool | None = None,
    tracker_transform=None,
) -> tuple[list[ViolationEvent], int, float]:
    """Run the core rule pipeline on one clip; returns (events, frames, fps).

    `stitch` overrides the config's stitch_enabled; `tracker_transform` wraps
    the base tracker before stitching (used by the dropout robustness study).
    """
    scene = load_scene_config(clip.scene)
    tracker_cfg = load_tracker_config(tracker_config_path)
    tracker = Tracker(tracker_cfg, weights=str(clip.weights), device=device, work_dir=work_dir)
    if tracker_transform is not None:
        tracker = tracker_transform(tracker)
    if stitch is None:
        stitch = tracker_cfg.stitch_enabled
    if stitch:
        from tvd.track.stitch import TrackStitcher

        tracker = TrackStitcher(
            tracker,
            max_gap_frames=tracker_cfg.stitch_max_gap_frames,
            max_dist_px=tracker_cfg.stitch_max_dist_px,
        )
    smoother = SignalSmoother(window=5)

    cap = cv2.VideoCapture(str(clip.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {clip.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    gap_tolerance = tracker_cfg.stitch_max_gap_frames if stitch else 1
    engine = RuleEngine(
        scene,
        fps=fps,
        min_track_duration=tracker_cfg.min_track_duration,
        max_track_gap=gap_tolerance,
    )

    frame_idx = 0
    ok, frame = cap.read()
    while ok:
        frame_idx += 1
        tracks = tracker.track_frame(frame, frame_idx)
        if scene.signal_roi is not None:
            engine.set_signal_state(frame_idx, smoother.update(frame, scene.signal_roi))
        engine.update(frame_idx, tracks)
        ok, frame = cap.read()
    cap.release()
    return list(engine.events), frame_idx, fps
