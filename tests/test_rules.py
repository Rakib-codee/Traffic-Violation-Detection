"""Rule tests on synthetic trajectories (plan section 7, phase 4)."""

import pytest

from tvd.config import Lane, Point, SceneConfig, SignalRoi
from tvd.rules import RuleEngine, ViolationType
from tvd.signal import SignalState
from tvd.track.tracker import Track

FPS = 10.0
TOLERANCE_S = 0.3  # -> 3 frames grace after signal turns red
WRONG_WAY_MIN_FRAMES = 10


@pytest.fixture
def scene() -> SceneConfig:
    return SceneConfig(
        camera_id="test",
        stop_line=[Point(x=0, y=100), Point(x=200, y=100)],  # horizontal
        lanes=[
            Lane(
                name="approach",
                polygon=[
                    Point(x=0, y=0),
                    Point(x=200, y=0),
                    Point(x=200, y=400),
                    Point(x=0, y=400),
                ],
                direction=Point(x=0, y=1),
            )
        ],
        signal_roi=SignalRoi(x=0, y=0, width=10, height=10),
        red_light_tolerance_s=TOLERANCE_S,
        wrong_way_min_frames=WRONG_WAY_MIN_FRAMES,
    )


def make_track(track_id: int, x: float, y_foot: float, frame: int, size: float = 10.0) -> Track:
    """Track with its ground reference (foot) at (x, y_foot)."""
    return Track(
        track_id=track_id,
        x1=x - size / 2,
        y1=y_foot - size * 2,
        x2=x + size / 2,
        y2=y_foot,
        confidence=0.9,
        class_name="car",
        frame_idx=frame,
    )


def drive(
    scene,
    frames: int,
    path,
    signal: SignalState = SignalState.RED,
    signal_from=1,
    tid: int = 1,
    min_track_duration: int = 5,
):
    """path: list of (frame, x, y_foot). Feeds frames to a RuleEngine."""
    engine = RuleEngine(scene, fps=FPS, min_track_duration=min_track_duration)
    points = {f: (x, y) for f, x, y in path}
    events = []
    for frame in range(1, frames + 1):
        engine.set_signal_state(frame, signal if frame >= signal_from else SignalState.GREEN)
        if frame in points:
            x, y = points[frame]
            events.extend(engine.update(frame, [make_track(tid, x, y, frame)]))
    return events, engine


def approaching_path(speed: float, frames: int = 60):
    """Vehicle moving +y (with lane direction), crossing y=100 mid-way."""
    start_y = 20.0
    return [(f, 100.0, start_y + speed * f) for f in range(1, frames + 1)]


# --- red-light rule ---


def test_red_light_crossing_flags(scene):
    path = approaching_path(speed=4.0)  # crosses y=100 at frame 20
    events, _ = drive(scene, 60, path, signal=SignalState.RED, signal_from=1)
    red = [e for e in events if e.violation_type is ViolationType.RED_LIGHT]
    assert len(red) == 1
    assert red[0].track_id == 1


def test_green_light_crossing_ok(scene):
    events, _ = drive(scene, 60, approaching_path(speed=4.0), signal=SignalState.GREEN)
    assert not [e for e in events if e.violation_type is ViolationType.RED_LIGHT]


def test_tolerance_window_suppresses_early_crossing(scene):
    # signal turns red at frame 19; vehicle crosses y=100 at frame 20 (1 frame later)
    events, _ = drive(scene, 60, approaching_path(speed=4.0), signal_from=19)
    assert not [e for e in events if e.violation_type is ViolationType.RED_LIGHT]


def test_crossing_after_tolerance_flags(scene):
    # signal turns red at frame 15; vehicle crosses at frame 20 (5 frames > 3 grace)
    events, _ = drive(scene, 60, approaching_path(speed=4.0), signal_from=15)
    assert len([e for e in events if e.violation_type is ViolationType.RED_LIGHT]) == 1


def test_no_duplicate_red_light_events(scene):
    path = [(f, 100.0, 120.0) for f in range(30, 60)] + approaching_path(4.0)[:29]
    events, _ = drive(scene, 60, sorted(path), signal_from=1)
    # vehicle already past the line oscillating should not re-flag
    assert len([e for e in events if e.violation_type is ViolationType.RED_LIGHT]) <= 1


# --- wrong-way rule ---


def test_wrong_way_flags(scene):
    # vehicle moving -y inside the lane (against direction +y)
    path = [(f, 100.0, 300.0 - 5 * f) for f in range(1, 61)]
    events, _ = drive(scene, 60, path, signal=SignalState.GREEN)
    wrong = [e for e in events if e.violation_type is ViolationType.WRONG_WAY]
    assert len(wrong) == 1
    assert wrong[0].end_frame >= WRONG_WAY_MIN_FRAMES + 1


def test_correct_direction_no_event(scene):
    events, _ = drive(scene, 60, approaching_path(speed=4.0), signal=SignalState.GREEN)
    assert not [e for e in events if e.violation_type is ViolationType.WRONG_WAY]


def test_stopped_vehicle_no_wrong_way(scene):
    path = [(f, 100.0, 200.0 + (0.05 * f)) for f in range(1, 61)]  # ~static
    events, _ = drive(scene, 60, path, signal=SignalState.GREEN)
    assert not events


def test_short_opposing_motion_no_event(scene):
    # wrong direction for only 4 frames (below min duration), then correct
    path = [(f, 100.0, 300.0 - 5 * f) for f in range(1, 5)]
    path += [(f, 100.0, 200.0 + 5 * f) for f in range(5, 40)]
    events, _ = drive(scene, 60, sorted(path), signal=SignalState.GREEN)
    assert not [e for e in events if e.violation_type is ViolationType.WRONG_WAY]


def test_occlusion_gap_then_wrong_way(scene):
    # track disappears frames 5-8, then continues moving the wrong way
    path = [(f, 100.0, 300.0 - 5 * f) for f in list(range(1, 5)) + list(range(9, 30))]
    events, _ = drive(scene, 60, path, signal=SignalState.GREEN)
    wrong = [e for e in events if e.violation_type is ViolationType.WRONG_WAY]
    # run length resets after the gap; frames 9+ give a long enough run again
    assert len(wrong) == 1
    assert wrong[0].start_frame >= 9
