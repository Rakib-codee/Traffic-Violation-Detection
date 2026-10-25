"""Build derived clips for the event-level eval set into data/events/.

Everything produced here is a *constructed* test case built from real footage,
documented as such in configs/event_set.yaml and docs/DATA.md:

- reverse: plays a clip backwards, so every moving vehicle drives against the
  lane direction by construction (wrong-way ground truth).
- signal overlay: draws a synthetic signal lamp in a fixed corner (green for
  the first `green_until` frames, then red), so stop-line crossings during
  the red window are red-light violations by construction.

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


def overlay_signal(src: Path, dst: Path, green_until: int) -> int:
    """Copy src with a synthetic signal lamp drawn in the top-left corner.

    The lamp is green through frame `green_until`, then red. The lamp area is
    fully saturated so the ROI color heuristic classifies it unambiguously;
    the matching scene config must point signal_roi at the lamp (see
    configs/scene.detrac-mvi20012-signal.yaml).
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_idx = 0
    ok, frame = cap.read()
    while ok:
        frame_idx += 1
        color = (0, 255, 0) if frame_idx <= green_until else (0, 0, 255)  # BGR
        cv2.rectangle(frame, (8, 8), (54, 94), (40, 40, 40), -1)  # housing
        cv2.circle(frame, (31, 30), 16, color, -1)  # active lamp
        cv2.circle(frame, (31, 70), 16, (60, 60, 60), -1)  # dark second lamp
        writer.write(frame)
        ok, frame = cap.read()
    cap.release()
    writer.release()
    logger.info("wrote %d frames (signal overlay, green<=%d) to %s", frame_idx, green_until, dst)
    return frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", default="data/samples")
    parser.add_argument("--out-dir", default="data/events")
    args = parser.parse_args()

    samples = Path(args.samples_dir)
    out_dir = Path(args.out_dir)

    # wrong-way by construction: all moving traffic opposes the lane direction
    reverse_video(samples / "detrac_mvi20012.mp4", out_dir / "detrac_mvi20012_reversed.mp4")

    # red-light by construction: green through frame 25, red afterwards; the
    # only stop-line crossing in the red window is the car at frame ~40
    overlay_signal(samples / "detrac_mvi20012.mp4", out_dir / "detrac_mvi20012_signal.mp4", 25)


if __name__ == "__main__":
    main()
