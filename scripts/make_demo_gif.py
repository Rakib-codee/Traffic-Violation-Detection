"""Render the README demo GIF from a tracked-video annotated mp4.

Input is the gitignored output of scripts/run_video.py (e.g.
results/tracking/detrac-demo/annotated.mp4); output is a small palette GIF
plus a poster frame under docs/assets/ (committed, kept small on purpose).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image

from tvd.utils import get_logger

logger = get_logger(__name__)


def make_gif(
    src: Path, gif_path: Path, poster_path: Path, width: int, fps: int, colors: int
) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {src}")
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise SystemExit(f"{src}: no frames read")

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    poster_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        ratio = width / img.width
        img = img.resize((width, round(img.height * ratio)), Image.LANCZOS)
        images.append(img.convert("P", palette=Image.ADAPTIVE, colors=colors))

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
    )
    mid = images[len(images) // 2].convert("RGB")
    mid.save(poster_path, quality=88)
    logger.info(
        "wrote %s (%d frames, %.1f KB) and %s",
        gif_path,
        len(images),
        gif_path.stat().st_size / 1024,
        poster_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="results/tracking/detrac-demo/annotated.mp4")
    parser.add_argument("--gif", default="docs/assets/demo-tracking.gif")
    parser.add_argument("--poster", default="docs/assets/demo-tracking.jpg")
    parser.add_argument("--width", type=int, default=280)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--colors", type=int, default=96)
    args = parser.parse_args()
    make_gif(Path(args.video), Path(args.gif), Path(args.poster), args.width, args.fps, args.colors)


if __name__ == "__main__":
    main()
