"""Track vehicles in a video: demo pipeline (detect -> track -> export).

Outputs in results/tracking/<run_name>/:
  tracks.txt        MOT-format tracks
  trajectories.csv  per-frame track records
  annotated.mp4     video with boxes and track IDs
  summary.json      frame count, unique IDs, FPS
"""

from __future__ import annotations

import argparse

from tvd.config import load_tracker_config
from tvd.track.pipeline import track_video
from tvd.track.tracker import Tracker
from tvd.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", default="weights/detector.pt")
    parser.add_argument("--tracker-config", default="configs/tracker.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--no-annotate", action="store_true")
    args = parser.parse_args()

    cfg = load_tracker_config(args.tracker_config)
    tracker = Tracker(cfg, weights=args.weights, device=args.device)
    summary = track_video(
        args.video,
        tracker,
        f"results/tracking/{args.run_name}",
        cfg,
        args.weights,
        annotate=not args.no_annotate,
    )
    print({k: summary[k] for k in ("frames", "unique_track_ids", "fps")})


if __name__ == "__main__":
    main()
