"""Assemble the LinkedIn/showcase demo video from real pipeline outputs.

Structure: title card -> tracking overlay clip -> wrong-way evidence clip ->
red-light evidence clip -> metrics card. All inputs are regenerated pipeline
outputs (gitignored); the output is H.264 (browser/LinkedIn compatible) and
lands in results/showcase/ (also gitignored). Evidence clips are already
blurred by the pipeline's default privacy step.

Run (after scripts/run_video.py and scripts/run_pipeline.py showcase runs):
    python scripts/make_showcase_video.py
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

from tvd.utils import get_logger
from tvd.utils.video import open_video_writer

logger = get_logger(__name__)

SIZE = (640, 640)
FPS = 10

BG = (24, 24, 24)
FG = (240, 240, 240)
ACCENT = (80, 220, 100)


def _text_center(frame, text, y, scale=1.0, color=FG, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(frame, text, ((SIZE[0] - tw) // 2, y), font, scale, color, thickness, cv2.LINE_AA)


def card(lines: list[tuple[str, float, tuple[int, int, int]]], seconds: float) -> list[np.ndarray]:
    """A solid title/metrics card; lines are (text, scale, color)."""
    frame = np.full((SIZE[1], SIZE[0], 3), BG, dtype=np.uint8)
    y0 = SIZE[1] // 2 - 40 * len(lines)
    for i, (text, scale, color) in enumerate(lines):
        _text_center(frame, text, y0 + i * 70, scale=scale, color=color)
    return [frame] * round(seconds * FPS)


def read_clip(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path} (generate it first; see docstring)")
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    return frames


def with_caption(frames: list[np.ndarray], caption: str) -> list[np.ndarray]:
    """Burn a caption bar into every frame of a segment."""
    out = []
    for frame in frames:
        f = frame.copy()
        cv2.rectangle(f, (0, 0), (SIZE[0], 34), (0, 0, 0), -1)
        cv2.putText(f, caption, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FG, 1, cv2.LINE_AA)
        out.append(f)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", default="results/tracking/detrac-demo/annotated.mp4")
    parser.add_argument(
        "--wrong-way-glob",
        default="results/pipeline/showcase-wrongway/evidence/*/clip.mp4",
    )
    parser.add_argument(
        "--red-light-glob",
        default="results/pipeline/showcase-redlight/evidence/*/clip.mp4",
    )
    parser.add_argument("--out", default="results/showcase/demo.mp4")
    args = parser.parse_args()

    wrong_way = sorted(glob.glob(args.wrong_way_glob))
    red_light = sorted(glob.glob(args.red_light_glob))
    if not wrong_way or not red_light:
        raise SystemExit("evidence clips not found; run the showcase pipelines first")

    segments = [
        card(
            [
                ("Traffic Violation Detection", 1.1, ACCENT),
                ("detect - track - scene rules - reviewable evidence", 0.6, FG),
            ],
            4,
        ),
        with_caption(read_clip(Path(args.tracking)), "YOLO + ByteTrack on UA-DETRAC sample"),
        with_caption(
            read_clip(Path(wrong_way[0])),
            "Flagged: wrong-way driving (constructed test clip)",
        ),
        with_caption(
            read_clip(Path(red_light[0])),
            "Flagged: red-light run (synthetic signal overlay)",
        ),
        card(
            [
                ("mAP@0.5 0.970  |  event P/R/F1 1.000", 0.8, FG),
                ("stitching: recall 0.33 -> 0.96 under dropout", 0.7, FG),
                ("github.com/Rakib-codee/Traffic-Violation-Detection", 0.55, ACCENT),
            ],
            5,
        ),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(str(out_path), FPS, SIZE)
    n = 0
    for segment in segments:
        for frame in segment:
            writer.write(frame)
            n += 1
    writer.release()
    logger.info("wrote %d frames (%.1fs) to %s", n, n / FPS, out_path)


if __name__ == "__main__":
    main()
