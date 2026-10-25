"""Video writer helper: prefer H.264 so clips play in browsers/players.

OpenCV's default mp4v (MPEG-4 Part 2) is not decodable by HTML5 video
players, so dashboard evidence clips and demo videos must be H.264 (avc1).
Falls back to mp4v with a warning when H.264 is unavailable.
"""

from __future__ import annotations

import cv2

from tvd.utils.logging import get_logger

logger = get_logger(__name__)


def open_video_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open a VideoWriter with H.264, falling back to mp4v if unsupported."""
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            if fourcc != "avc1":
                logger.warning(
                    "H.264 unavailable; %s written with %s (may not play in browsers)",
                    path,
                    fourcc,
                )
            return writer
    raise ValueError(f"cannot open video writer for {path}")
