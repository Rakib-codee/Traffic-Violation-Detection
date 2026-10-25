"""Labeled synthetic scenarios for event-level rule evaluation.

Each scenario scripts vehicle trajectories + a signal timeline, declares the
violations that *should* fire, and drives a RuleEngine. This is the evaluation
until the hand-labeled event-level test set exists (see docs/DATA.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from tvd.config import Lane, Point, SceneConfig, SignalRoi
from tvd.rules import RuleEngine, ViolationType
from tvd.signal import SignalState
from tvd.track.tracker import Track


@dataclass
class Scenario:
    name: str
    frames: int
    # list of trajectories: (track_id, [(frame, x_foot, y_foot), ...])
    trajectories: list[tuple[int, list[tuple[int, float, float]]]]
    signal_from_frame: int = 1
    signal_before: SignalState = SignalState.GREEN
    signal_after: SignalState = SignalState.RED
    expected: list[tuple[int, ViolationType]] | None = None  # (track_id, type)


def default_scene() -> SceneConfig:
    return SceneConfig(
        camera_id="scenario",
        stop_line=[Point(x=0, y=100), Point(x=200, y=100)],
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
    )


def make_track(track_id: int, x: float, y_foot: float, frame: int, size: float = 10.0) -> Track:
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


def run_scenario(scenario: Scenario, scene: SceneConfig | None = None, fps: float = 10.0):
    """Drive the rule engine over the scenario; returns (predicted, expected)."""
    scene = scene or default_scene()
    engine = RuleEngine(scene, fps=fps)
    by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for track_id, points in scenario.trajectories:
        for frame, x, y in points:
            by_frame.setdefault(frame, []).append((track_id, x, y))

    predicted: list[tuple[int, ViolationType]] = []
    for frame in range(1, scenario.frames + 1):
        state = (
            scenario.signal_after if frame >= scenario.signal_from_frame else scenario.signal_before
        )
        engine.set_signal_state(frame, state)
        tracks = [make_track(tid, x, y, frame) for tid, x, y in by_frame.get(frame, [])]
        predicted.extend((e.track_id, e.violation_type) for e in engine.update(frame, tracks))
    return predicted, scenario.expected or []


def approach(speed: float, frames: int = 60, start_y: float = 20.0, x: float = 100.0):
    """Vehicle moving with the lane (+y), crossing y=100 at (100-start_y)/speed."""
    return [(f, x, start_y + speed * f) for f in range(1, frames + 1)]


def wrong_way(speed: float = 5.0, frames: int = 60, start_y: float = 300.0, x: float = 100.0):
    return [(f, x, start_y - speed * f) for f in range(1, frames + 1)]


def stationary(y: float = 200.0, frames: int = 60, x: float = 100.0):
    return [(f, x, y + 0.05 * f) for f in range(1, frames + 1)]


def occlusion_recross(x: float = 100.0):
    """Cross on green, hover past the line, bbox-truncate back across it during
    'occlusion', then re-expand past the line on red. The re-cross is an
    artifact and must be suppressed by the recross window."""
    points = [(f, x, 60.0 + 6.0 * f) for f in range(1, 16)]  # crosses y=100 at f7 (green)
    points += [(f, x, 125.0) for f in range(16, 21)]  # genuinely past the line
    points += [(f, x, 95.0) for f in range(21, 55)]  # truncated during occlusion
    points += [(55, x, 140.0)]  # occlusion clears: re-cross on red (artifact)
    return points


def recross_after_window(x: float = 100.0):
    """Cross on green, retreat far past the window, then cross again on red
    after the recross window has expired — a genuine second crossing."""
    points = [(f, x, 60.0 + 6.0 * f) for f in range(1, 16)]  # crosses at f7 (green)
    points += [(f, x, 125.0) for f in range(16, 21)]
    points += [(f, x, 95.0) for f in range(21, 80)]  # back before the line
    points += [(f, x, 95.0 + 10.0 * (f - 79)) for f in range(80, 82)]  # re-cross at f81
    return points


SCENARIOS: list[Scenario] = [
    Scenario(
        name="red_light_runner",
        frames=60,
        trajectories=[(1, approach(4.0))],
        signal_from_frame=1,
        expected=[(1, ViolationType.RED_LIGHT)],
    ),
    Scenario(
        name="legal_green_crossing",
        frames=60,
        trajectories=[(1, approach(4.0))],
        signal_from_frame=1,
        signal_after=SignalState.GREEN,
        expected=[],
    ),
    Scenario(
        name="crossing_in_tolerance_window",
        frames=60,
        trajectories=[(1, approach(4.0))],  # crosses at frame 20
        signal_from_frame=19,  # red 1 frame before crossing -> grace
        expected=[],
    ),
    Scenario(
        name="wrong_way_driver",
        frames=60,
        trajectories=[(1, wrong_way())],
        signal_from_frame=1,
        signal_after=SignalState.GREEN,
        expected=[(1, ViolationType.WRONG_WAY)],
    ),
    Scenario(
        name="normal_two_directions",
        frames=60,
        trajectories=[(1, approach(4.0)), (2, approach(4.0, x=140.0))],
        signal_from_frame=1,
        signal_after=SignalState.GREEN,
        expected=[],
    ),
    Scenario(
        name="one_runner_one_legal",
        frames=60,
        trajectories=[(1, approach(4.0)), (2, approach(4.0, x=140.0))],
        signal_from_frame=1,
        expected=[(1, ViolationType.RED_LIGHT), (2, ViolationType.RED_LIGHT)],
    ),
    Scenario(
        name="stopped_vehicle_on_red",
        frames=60,
        trajectories=[(1, stationary())],
        signal_from_frame=1,
        expected=[],
    ),
    Scenario(
        name="occlusion_recross_on_red",
        frames=60,
        trajectories=[(1, occlusion_recross())],
        signal_from_frame=50,  # green until frame 49, red from 50
        expected=[],  # re-cross at f55 is a bbox artifact; suppressed
    ),
    Scenario(
        name="recross_after_window_fires",
        frames=90,
        trajectories=[(1, recross_after_window())],
        signal_from_frame=50,
        expected=[(1, ViolationType.RED_LIGHT)],  # genuine re-cross at f81 (>window)
    ),
]
