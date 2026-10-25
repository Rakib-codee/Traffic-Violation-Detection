"""Build derived clips for the event-level eval set into data/events/.

Everything produced here is a *constructed* test case built from real footage,
documented as such in configs/event_set.yaml and docs/DATA.md:

- reverse: plays a clip backwards, so every moving vehicle drives against the
  lane direction by construction (wrong-way ground truth).

Outputs land in data/events/ (gitignored). Re-running is deterministic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from tvd.utils import get_logger

logger = get_logger(__name__)


def reverse_video(src: Path, dst: Path) -> int:
    """Write src played backwards to dst; returns the frame count."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise SystemExit(f"{src}: no frames read")

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in reversed(frames):
        writer.write(frame)
    writer.release()
    logger.info("wrote %d frames (reversed) to %s", len(frames), dst)
    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", default="data/samples")
    parser.add_argument("--out-dir", default="data/events")
    args = parser.parse_args()

    samples = Path(args.samples_dir)
    out_dir = Path(args.out_dir)

    # wrong-way by construction: all moving traffic opposes the lane direction
    reverse_video(samples / "detrac_mvi20012.mp4", out_dir / "detrac_mvi20012_reversed.mp4")


if __name__ == "__main__":
    main()
