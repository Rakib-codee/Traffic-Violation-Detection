"""Deterministic degradation transforms for robustness evaluation.

Night, rain, and glare variants used to re-evaluate the detector on an
augmented validation split (scripts/eval_robustness.py). Seeded via the
system config so runs are reproducible.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

TransformName = Literal["night", "rain", "glare"]


def apply_night(frame: np.ndarray) -> np.ndarray:
    """Darken + desaturate + blue shift, like a poorly lit night scene."""
    out = frame.astype(np.float32) / 255.0
    out = np.power(out, 1.8)  # gamma darken
    hsv = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.75
    hsv[..., 2] *= 0.75
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_rain(frame: np.ndarray, rng: np.random.Generator, n_streaks: int = 120) -> np.ndarray:
    """Diagonal streaks + slight haze."""
    out = frame.copy()
    h, w = frame.shape[:2]
    for _ in range(n_streaks):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        length = int(rng.integers(8, 25))
        cv2.line(out, (x, y), (x + 2, y + length), (200, 200, 210), 1)
    return cv2.addWeighted(out, 0.9, np.full_like(out, 15), 0.1, 0)


def apply_glare(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A bright light source with a soft glow."""
    out = frame.astype(np.float32)
    h, w = frame.shape[:2]
    cx, cy = int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(0, h // 3))
    radius = int(rng.integers(30, 70))
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    glow = np.clip(1 - dist / radius, 0, 1) ** 2
    out += glow[..., None] * 220
    return np.clip(out, 0, 255).astype(np.uint8)


TRANSFORMS: dict[TransformName, callable] = {
    "night": lambda frame, rng: apply_night(frame),
    "rain": apply_rain,
    "glare": apply_glare,
}


def augment_frame(frame: np.ndarray, kind: TransformName, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return TRANSFORMS[kind](frame, rng)
