"""Traffic-signal state classification.

Phase 3 baseline is a color heuristic (plan section 5.4): HSV threshold
counts inside the signal ROI, picked by a small CNN later if needed.
Temporal smoothing is a majority vote over a sliding window.
"""

from __future__ import annotations

from collections import deque
from enum import Enum

import cv2
import numpy as np

from tvd.config import SignalRoi
from tvd.utils import get_logger

logger = get_logger(__name__)


class SignalState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


# OpenCV HSV hue is 0..179
_RANGES: dict[SignalState, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    SignalState.RED: [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (179, 255, 255))],
    SignalState.YELLOW: [((15, 100, 100), (40, 255, 255))],
    SignalState.GREEN: [((40, 80, 80), (90, 255, 255))],
}

_MIN_PIXELS = 30


def classify_roi(frame: np.ndarray, roi: SignalRoi) -> SignalState:
    """Classify the signal state within an ROI of a BGR frame."""
    h, w = frame.shape[:2]
    x1, y1 = max(int(roi.x), 0), max(int(roi.y), 0)
    x2, y2 = min(int(roi.x + roi.width), w), min(int(roi.y + roi.height), h)
    if x2 <= x1 or y2 <= y1:
        return SignalState.UNKNOWN

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    best_state, best_count = SignalState.UNKNOWN, _MIN_PIXELS
    for state, ranges in _RANGES.items():
        count = 0
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            count += int(cv2.countNonZero(mask))
        if count > best_count:
            best_state, best_count = state, count
    return best_state


class SignalSmoother:
    """Majority-vote smoothing over the last N classified frames."""

    def __init__(self, window: int = 5):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._history: deque[SignalState] = deque(maxlen=window)

    def update(self, frame: np.ndarray, roi: SignalRoi) -> SignalState:
        self._history.append(classify_roi(frame, roi))
        votes: dict[SignalState, int] = {}
        for state in self._history:
            if state is not SignalState.UNKNOWN:
                votes[state] = votes.get(state, 0) + 1
        if not votes:
            return SignalState.UNKNOWN
        return max(votes, key=votes.get)
