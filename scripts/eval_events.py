"""Event-level evaluation on the labeled real-video event set.

Runs the full pipeline (detect -> track -> signal -> rules) on each clip in
configs/event_set.yaml and matches predicted violation events against the
hand-labeled expected events by type and frame-window overlap
(±match_tolerance_frames). Writes results/events/metrics.json with
precision/recall/F1 and false alerts per hour of video.

Clip videos are gitignored; build derived clips first:
    python scripts/build_event_set.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tvd.config import load_event_set_config
from tvd.pipeline import run_clip
from tvd.rules import EventWindow, match_events
from tvd.utils import get_logger

logger = get_logger(__name__)


def evaluate(
    manifest_path: Path = Path("configs/event_set.yaml"),
    tracker_config: str = "configs/tracker.yaml",
    out_dir: Path = Path("results/events"),
    device: str = "auto",
) -> dict:
    manifest = load_event_set_config(manifest_path)
    tolerance = manifest.match_tolerance_frames
    tp = fp = fn = 0
    total_seconds = 0.0
    rows = []

    for clip in manifest.clips:
        events, n_frames, fps = run_clip(clip, tracker_config, device, out_dir / "work")
        predicted = [
            EventWindow(e.violation_type.value, e.start_frame, e.end_frame) for e in events
        ]
        expected = [EventWindow(e.type, e.start_frame, e.end_frame) for e in clip.expected_events]
        c_tp, c_fp, c_fn = match_events(predicted, expected, tolerance)
        tp, fp, fn = tp + c_tp, fp + c_fp, fn + c_fn
        total_seconds += n_frames / fps
        rows.append(
            {
                "clip": clip.name,
                "frames": n_frames,
                "tp": c_tp,
                "fp": c_fp,
                "fn": c_fn,
                "predicted": [(e.type, e.start_frame, e.end_frame) for e in predicted],
                "expected": [(e.type, e.start_frame, e.end_frame) for e in expected],
            }
        )
        status = "OK" if c_fp == 0 and c_fn == 0 else "MISMATCH"
        logger.info("%s: %s (tp=%d fp=%d fn=%d)", clip.name, status, c_tp, c_fp, c_fn)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_alerts_per_hour = fp / (total_seconds / 3600.0) if total_seconds else 0.0

    result = {
        "note": "real-video event set; see configs/event_set.yaml for labeling method",
        "manifest": str(manifest_path),
        "match_tolerance_frames": tolerance,
        "n_clips": len(manifest.clips),
        "video_seconds": round(total_seconds, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alerts_per_hour": round(false_alerts_per_hour, 2),
        "clips": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    logger.info(
        "event-level: precision=%.3f recall=%.3f f1=%.3f false-alerts/h=%.1f",
        precision,
        recall,
        f1,
        false_alerts_per_hour,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/event_set.yaml")
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--out-dir", default="results/events")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(Path(args.manifest), args.tracker_config, Path(args.out_dir), args.device)


if __name__ == "__main__":
    main()
