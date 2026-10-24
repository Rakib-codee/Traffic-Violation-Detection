"""End-to-end pipeline: video -> track -> signal -> rules -> evidence -> DB.

Every produced event is a candidate for human review; plates/faces are
blurred in stored evidence by default (disable with --no-blur).

Phase 6 extensions from configs/extensions.yaml run alongside the core rules:
the helmet rule crops motorcycle tracks through a secondary detector.

Example:
    python scripts/run_pipeline.py --video data/samples/synthetic_intersection.mp4 \
        --scene configs/scene.example.yaml --weights weights/detector.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from tvd.config import load_extensions_config, load_scene_config, load_tracker_config
from tvd.evidence import EventStore, EvidenceBuilder
from tvd.rules import RuleEngine
from tvd.signal import SignalSmoother
from tvd.track.tracker import Tracker
from tvd.utils import get_logger

logger = get_logger(__name__)


def run_pipeline(
    video: str,
    scene_config: str,
    weights: str = "weights/detector.pt",
    tracker_config: str = "configs/tracker.yaml",
    extensions_config: str = "configs/extensions.yaml",
    out_dir: str = "results/pipeline",
    db_path: str = "results/events.db",
    device: str = "auto",
    blur: bool = True,
    clip_seconds: float = 7.0,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = load_scene_config(scene_config)
    tracker_cfg = load_tracker_config(tracker_config)
    tracker = Tracker(tracker_cfg, weights=weights, device=device, work_dir=out_dir)

    ext = load_extensions_config(extensions_config)
    helmet_rule = None
    if ext.helmet.enabled:
        from tvd.rules.helmet import HelmetRule, make_yolo_crop_detector

        helmet_rule = HelmetRule(
            ext.helmet,
            make_yolo_crop_detector(ext.helmet, device=device),
            camera_id=scene.camera_id,
        )
        logger.info("helmet rule enabled (weights=%s)", ext.helmet.weights)
    helmet_events = []

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    engine = RuleEngine(scene, fps=fps, min_track_duration=tracker_cfg.min_track_duration)
    smoother = SignalSmoother(window=5)
    boxes_by_frame: dict[int, list[tuple[int, int, int, int]]] = {}
    tracks_by_frame: dict[int, list] = {}

    frame_idx = 0
    ok, frame = cap.read()
    while ok:
        frame_idx += 1
        tracks = tracker.track_frame(frame, frame_idx)
        tracks_by_frame[frame_idx] = tracks
        boxes_by_frame[frame_idx] = [(int(t.x1), int(t.y1), int(t.x2), int(t.y2)) for t in tracks]
        if scene.signal_roi is not None:
            engine.set_signal_state(frame_idx, smoother.update(frame, scene.signal_roi))
        engine.update(frame_idx, tracks)
        if helmet_rule is not None:
            for track in tracks:
                event = helmet_rule.update(frame, frame_idx, track)
                if event is not None:
                    helmet_events.append(event)
        ok, frame = cap.read()
    cap.release()

    builder = EvidenceBuilder(
        video,
        out_dir / "evidence",
        clip_seconds=clip_seconds,
        blur=blur,
        vehicle_boxes_by_frame=boxes_by_frame,
    )
    store = EventStore(db_path)
    all_events = list(engine.events) + helmet_events
    for event in all_events:
        key_tracks = tracks_by_frame.get((event.start_frame + event.end_frame) // 2, [])
        item = builder.build(event, keyframe_tracks=key_tracks)
        store.add_event(event, item)

    summary = {
        "video": str(video),
        "scene": scene.camera_id,
        "frames": frame_idx,
        "tracked_detections": sum(len(t) for t in tracks_by_frame.values()),
        "events": len(all_events),
        "event_types": sorted({e.violation_type.value for e in all_events}),
        "helmet_enabled": helmet_rule is not None,
        "blur": blur,
        "db": db_path,
    }
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("pipeline done: %d frames, %d events", frame_idx, len(all_events))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--scene", required=True, help="configs/scene.<camera>.yaml")
    parser.add_argument("--weights", default="weights/detector.pt")
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--extensions-config", default="configs/extensions.yaml")
    parser.add_argument("--out-dir", default="results/pipeline")
    parser.add_argument("--db", default="results/events.db")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clip-seconds", type=float, default=7.0)
    parser.add_argument("--no-blur", action="store_true")
    args = parser.parse_args()
    run_pipeline(
        args.video,
        args.scene,
        weights=args.weights,
        tracker_config=args.tracker_config,
        extensions_config=args.extensions_config,
        out_dir=args.out_dir,
        db_path=args.db,
        device=args.device,
        blur=not args.no_blur,
        clip_seconds=args.clip_seconds,
    )


if __name__ == "__main__":
    main()
