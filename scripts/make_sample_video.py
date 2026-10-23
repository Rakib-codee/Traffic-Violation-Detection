"""Generate a small synthetic traffic video for smoke tests and demos.

Draws a stop line and rectangular 'vehicles' moving along two lanes.
Deterministic for a given seed. Output goes to the gitignored data/ dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tvd.config import load_system_config
from tvd.utils import get_logger, set_seed

logger = get_logger(__name__)

WIDTH, HEIGHT, FPS, N_FRAMES = 640, 360, 10, 50
STOP_LINE_Y = 240


def draw_scene(frame: np.ndarray) -> None:
    """Static background: road, lane divider, stop line."""
    cv2.rectangle(frame, (180, 0), (460, HEIGHT), (60, 60, 60), -1)  # road
    cv2.line(frame, (320, 0), (320, HEIGHT), (200, 200, 200), 2)  # divider
    cv2.line(frame, (180, STOP_LINE_Y), (460, STOP_LINE_Y), (0, 0, 255), 3)  # stop line


def vehicle_positions(rng: np.random.Generator, n_frames: int) -> list[tuple[int, int, int, int]]:
    """Per-frame (x1, y1, x2, y2) for one southbound vehicle crossing the stop line."""
    speed = 6 + int(rng.integers(0, 3))
    x = 250 + int(rng.integers(-10, 10))
    w, h = 50, 90
    return [(x, y - h, x + w, y) for y in range(-h, n_frames * speed, speed)]


def generate(out_path: Path, seed: int) -> Path:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    positions = vehicle_positions(rng, N_FRAMES)
    for i in range(N_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 120, dtype=np.uint8)
        draw_scene(frame)
        if i < len(positions):
            x1, y1, x2, y2 = positions[i]
            cv2.rectangle(frame, (x1, max(y1, 0)), (x2, y2), (30, 90, 200), -1)
        writer.write(frame)
    writer.release()
    logger.info("wrote %d frames to %s", N_FRAMES, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/system.yaml")
    args = parser.parse_args()

    cfg = load_system_config(args.config)
    generate(cfg.sample_video, cfg.seed)


if __name__ == "__main__":
    main()
