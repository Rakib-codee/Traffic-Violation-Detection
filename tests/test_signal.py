import numpy as np
import pytest

from tvd.config import SignalRoi
from tvd.signal import SignalSmoother, SignalState, classify_roi

ROI = SignalRoi(x=0, y=0, width=40, height=90)


def _frame(color: tuple[int, int, int]) -> np.ndarray:
    """BGR frame filled with one color (plus dark background)."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[10:100, 10:50] = color
    return frame


def test_classifies_red():
    assert classify_roi(_frame((0, 0, 255)), ROI) is SignalState.RED


def test_classifies_yellow():
    assert classify_roi(_frame((0, 255, 255)), ROI) is SignalState.YELLOW


def test_classifies_green():
    assert classify_roi(_frame((0, 255, 0)), ROI) is SignalState.GREEN


def test_unknown_when_no_light():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert classify_roi(frame, ROI) is SignalState.UNKNOWN


def test_roi_outside_frame_is_unknown():
    frame = _frame((0, 0, 255))
    bad = SignalRoi(x=500, y=500, width=40, height=40)
    assert classify_roi(frame, bad) is SignalState.UNKNOWN


def test_smoother_majority_vote():
    smoother = SignalSmoother(window=3)
    green = _frame((0, 255, 0))
    red = _frame((0, 0, 255))
    assert smoother.update(green, ROI) is SignalState.GREEN
    # one noisy red frame should not flip the state yet
    assert smoother.update(red, ROI) is SignalState.GREEN
    assert smoother.update(red, ROI) is SignalState.RED


def test_smoother_all_unknown():
    smoother = SignalSmoother(window=2)
    dark = np.zeros((120, 160, 3), dtype=np.uint8)
    assert smoother.update(dark, ROI) is SignalState.UNKNOWN
    assert smoother.update(dark, ROI) is SignalState.UNKNOWN


def test_smoother_rejects_bad_window():
    with pytest.raises(ValueError):
        SignalSmoother(window=0)
