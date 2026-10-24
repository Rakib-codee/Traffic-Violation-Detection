"""Violation rule engine.

Consumes per-frame tracks, smoothed signal state, and the scene config.
Emits ViolationEvent objects; everything is a candidate for human review.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from tvd.config import Point, SceneConfig
from tvd.scene import crosses_stop_line, direction_alignment, lane_for_point
from tvd.signal import SignalState
from tvd.track.tracker import Track


class ViolationType(Enum):
    RED_LIGHT = "red_light"
    WRONG_WAY = "wrong_way"


@dataclass
class ViolationEvent:
    """A candidate violation for human review (never an automated penalty)."""

    violation_type: ViolationType
    track_id: int
    start_frame: int
    end_frame: int
    camera_id: str
    details: dict = field(default_factory=dict)


@dataclass
class _TrackState:
    """Rolling per-track history used by the rules."""

    last_center: tuple[float, float] | None = None
    last_frame: int = -1
    history: list[tuple[int, float, float]] = field(default_factory=list)
    red_light_flagged: bool = False
    wrong_way_flagged: bool = False
    wrong_way_run: int = 0
    wrong_way_start: int = -1


class RuleEngine:
    """Stateful per-frame rule evaluation. Feed tracks frame by frame."""

    def __init__(self, scene: SceneConfig, fps: float, min_track_duration: int = 5):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.scene = scene
        self.fps = fps
        self.min_track_duration = min_track_duration
        self._tracks: dict[int, _TrackState] = {}
        self._signal_history: dict[int, SignalState] = {}
        self.events: list[ViolationEvent] = []

    def set_signal_state(self, frame_idx: int, state: SignalState) -> None:
        self._signal_history[frame_idx] = state

    def signal_at(self, frame_idx: int) -> SignalState:
        """Latest known signal state at/before the frame."""
        known = [f for f in self._signal_history if f <= frame_idx]
        return self._signal_history[max(known)] if known else SignalState.UNKNOWN

    def update(self, frame_idx: int, tracks: list[Track]) -> list[ViolationEvent]:
        """Evaluate rules for one frame; returns events raised this frame."""
        new_events: list[ViolationEvent] = []
        for track in tracks:
            state = self._tracks.setdefault(track.track_id, _TrackState())
            cx, cy = track.center
            foot = ((track.x1 + track.x2) / 2, track.y2)  # ground reference point

            # occlusion gap: the tracker lost this id for >1 frame; reset the
            # wrong-way run so evidence comes from continuous observation
            if state.last_frame > 0 and frame_idx - state.last_frame > 1:
                state.wrong_way_run = 0

            if state.history:
                red = self._check_red_light(track, state, foot, frame_idx)
                if red:
                    new_events.append(red)
                wrong = self._check_wrong_way(track, state, frame_idx)
                if wrong:
                    new_events.append(wrong)

            state.history.append((frame_idx, foot[0], foot[1]))
            state.last_center = (cx, cy)
            state.last_frame = frame_idx

        self.events.extend(new_events)
        return new_events

    def _check_red_light(
        self, track: Track, state: _TrackState, foot: tuple[float, float], frame_idx: int
    ) -> ViolationEvent | None:
        if state.red_light_flagged or not state.history:
            return None
        prev_x, prev_y = state.history[-1][1], state.history[-1][2]
        prev = Point(x=prev_x, y=prev_y)
        curr = Point(x=foot[0], y=foot[1])
        if not crosses_stop_line(prev, curr, self.scene):
            return None

        signal = self.signal_at(frame_idx)
        if signal is not SignalState.RED:
            return None

        # tolerance window: earliest frame where the signal was already red
        tolerance_frames = self.scene.red_light_tolerance_s * self.fps
        red_start = frame_idx
        while (
            red_start - 1 in self._signal_history
            and self._signal_history[red_start - 1] is SignalState.RED
        ):
            red_start -= 1
        if frame_idx - red_start < tolerance_frames:
            return None  # just turned red; grace window

        state.red_light_flagged = True
        return ViolationEvent(
            violation_type=ViolationType.RED_LIGHT,
            track_id=track.track_id,
            start_frame=frame_idx,
            end_frame=frame_idx,
            camera_id=self.scene.camera_id,
            details={"cross_point": foot},
        )

    def _check_wrong_way(
        self, track: Track, state: _TrackState, frame_idx: int
    ) -> ViolationEvent | None:
        if state.wrong_way_flagged:
            return None
        foot_x = (track.x1 + track.x2) / 2
        foot_y = track.y2
        lane = lane_for_point(self.scene, foot_x, foot_y)
        if lane is None:
            state.wrong_way_run = 0
            return None

        prev_x, prev_y = state.history[-1][1], state.history[-1][2]
        motion = Point(x=foot_x - prev_x, y=foot_y - prev_y)
        if math.hypot(motion.x, motion.y) < 0.5:  # essentially stopped
            state.wrong_way_run = 0
            return None

        if direction_alignment(motion, lane) < 0:
            if state.wrong_way_run == 0:
                state.wrong_way_start = frame_idx
            state.wrong_way_run += 1
        else:
            state.wrong_way_run = 0

        min_frames = max(self.scene.wrong_way_min_frames, self.min_track_duration)
        if state.wrong_way_run >= min_frames:
            state.wrong_way_flagged = True
            return ViolationEvent(
                violation_type=ViolationType.WRONG_WAY,
                track_id=track.track_id,
                start_frame=state.wrong_way_start,
                end_frame=frame_idx,
                camera_id=self.scene.camera_id,
                details={"lane": lane.name},
            )
        return None
