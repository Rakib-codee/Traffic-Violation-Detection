"""Tracking evaluation: track a video, then score against MOT ground truth.

With --gt-folder (TrackEval MOTChallenge layout, e.g. produced by
convert_detrac_to_mot.py), writes results/tracking_eval/<run>/metrics.json
with HOTA, IDF1, ID switches, MOTA and prints the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tvd.config import load_tracker_config
from tvd.track.evaluate import run_trackeval, summarize
from tvd.track.mot import read_mot
from tvd.track.pipeline import track_video
from tvd.track.tracker import Tracker
from tvd.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="video whose name matches the GT sequence")
    parser.add_argument("--weights", default="weights/detector.pt")
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--gt-folder", default=None, help="TrackEval GT folder (MOT layout)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="eval")
    args = parser.parse_args()

    cfg = load_tracker_config(args.tracker_config)
    tracker = Tracker(cfg, weights=args.weights, device=args.device)
    run_dir = Path(f"results/tracking_eval/{args.run_name}")
    summary = track_video(args.video, tracker, run_dir / "tracks_run", cfg, args.weights)

    if not args.gt_folder:
        logger.info("no --gt-folder; tracked only, metrics not computed")
        print(json.dumps(summary, indent=2))
        return

    gt_folder = Path(args.gt_folder)
    seq_name = Path(args.video).stem
    if not (gt_folder / seq_name / "gt" / "gt.txt").exists():
        raise SystemExit(f"no GT for sequence '{seq_name}' under {gt_folder}")

    # TrackEval layout: trackers/<name>/data/<seq>.txt
    tracker_name = "tvd"
    pred_dir = run_dir / "trackers" / tracker_name / "data"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = read_mot(run_dir / "tracks_run" / "tracks.txt")
    (pred_dir / f"{seq_name}.txt").write_text("\n".join(r.to_line() for r in rows) + "\n")

    results = run_trackeval(gt_folder, run_dir / "trackers", [tracker_name])
    flat = summarize(results.get(tracker_name, {}))
    (run_dir / "metrics.json").write_text(
        json.dumps({"video": args.video, "summary": summary, "metrics": flat}, indent=2)
    )
    print(json.dumps(flat, indent=2))


if __name__ == "__main__":
    main()
