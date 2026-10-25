"""Match predicted violation events against labeled ground truth.

Used by the real-video event set (scripts/eval_events.py) and the dropout
robustness study (scripts/eval_dropout.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EventWindow:
    """A violation event as a type plus a 1-based frame window."""

    type: str
    start_frame: int
    end_frame: int


def windows_overlap(a: EventWindow, b: EventWindow, tolerance: int) -> bool:
    return a.start_frame <= b.end_frame + tolerance and b.start_frame <= a.end_frame + tolerance


def match_events(
    predicted: list[EventWindow], expected: list[EventWindow], tolerance: int
) -> tuple[int, int, int]:
    """Greedy one-to-one matching by type and ±tolerance frame overlap.

    Returns (tp, fp, fn). Each predicted event matches at most one expected
    event and vice versa.
    """
    expected_used = [False] * len(expected)
    tp = 0
    for pred in sorted(predicted, key=lambda e: e.start_frame):
        for i, exp in enumerate(expected):
            if expected_used[i] or exp.type != pred.type:
                continue
            if windows_overlap(pred, exp, tolerance):
                expected_used[i] = True
                tp += 1
                break
    fp = len(predicted) - tp
    fn = len(expected) - sum(expected_used)
    return tp, fp, fn
