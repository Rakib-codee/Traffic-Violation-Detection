"""Privacy blurring for stored evidence.

Plates and faces are blurred by default (plan section 5.8). Until Phase 6
adds real plate detection, plates are covered by a heuristic band across the
lower part of each vehicle box; person boxes are blurred at the top (head).
"""

from __future__ import annotations

import cv2
import numpy as np


def blur_regions(
    frame: np.ndarray, boxes: list[tuple[int, int, int, int]], strength: int = 25
) -> np.ndarray:
    """Gaussian-blur the given (x1, y1, x2, y2) boxes in-place-safe copy."""
    out = frame.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        region = out[y1:y2, x1:x2]
        out[y1:y2, x1:x2] = cv2.GaussianBlur(region, (strength | 1, strength | 1), 0)
    return out


def plate_band(box: tuple[int, int, int, int], frac: float = 0.2) -> tuple[int, int, int, int]:
    """Lower fraction of a vehicle box — the plate region (heuristic)."""
    x1, y1, x2, y2 = box
    band_h = max(int((y2 - y1) * frac), 4)
    return (x1, max(y2 - band_h, y1), x2, y2)


def head_band(box: tuple[int, int, int, int], frac: float = 0.3) -> tuple[int, int, int, int]:
    """Upper fraction of a person box — the head/face region (heuristic)."""
    x1, y1, x2, y2 = box
    band_h = max(int((y2 - y1) * frac), 4)
    return (x1, y1, x2, min(y1 + band_h, y2))


def blur_plate_band(
    frame: np.ndarray,
    vehicle_boxes: list[tuple[int, int, int, int]],
    person_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """Blur plate bands of vehicles and head bands of persons."""
    regions = [plate_band(b) for b in vehicle_boxes]
    regions += [head_band(b) for b in (person_boxes or [])]
    return blur_regions(frame, regions)
