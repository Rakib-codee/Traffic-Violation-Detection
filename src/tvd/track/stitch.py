"""Online tracklet stitching across short tracker gaps.

ByteTrack drops a track after its buffer expires and re-identifies the same
physical object with a new ID; occlusion gaps fragment one vehicle into
several tracklets. TrackStitcher wraps a base tracker and re-links a new
tracklet to a recently lost one when the position matches a constant-velocity
prediction. It is deliberately lightweight: geometry only, no appearance
model.

Gap handling note: stitched IDs alone do not restore rule-engine continuity
(the engine still sees no observation for the missing frames), so the rule
engine gets a matching per-track gap tolerance (RuleEngine max_track_gap)
fed from the same config value. No synthetic detections are fabricated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from tvd.track.tracker import Track, Tracker
from tvd.utils import get_logger

logger = get_logger(__name__)


def _foot(track: Track) -> tuple[float, float]:
    return ((track.x1 + track.x2) / 2, track.y2)


def _velocity(recent: list[Track]) -> tuple[float, float]:
    """Median per-frame foot displacement over consecutive-frame pairs."""
    dxs, dys = [], []
    for prev, curr in zip(recent, recent[1:], strict=False):
        if curr.frame_idx - prev.frame_idx != 1:
            continue
        px, py = _foot(prev)
        cx, cy = _foot(curr)
        dxs.append(cx - px)
        dys.append(cy - py)
    if not dxs:
        return (0.0, 0.0)
    return (float(np.median(dxs)), float(np.median(dys)))


@dataclass
class _LostTracklet:
    last: Track
    velocity: tuple[float, float]
    recent: list[Track]


class TrackStitcher:
    """Wraps a Tracker; re-links new tracklets to recently lost ones."""

    def __init__(
        self,
        base: Tracker,
        max_gap_frames: int = 10,
        max_dist_px: float = 80.0,
        velocity_window: int = 5,
    ):
        if max_gap_frames < 1:
            raise ValueError("max_gap_frames must be >= 1")
        if max_dist_px <= 0:
            raise ValueError("max_dist_px must be positive")
        self.base = base
        self.max_gap_frames = max_gap_frames
        self.max_dist_px = max_dist_px
        self.velocity_window = velocity_window
        self._active: dict[int, list[Track]] = {}
        self._lost: dict[int, _LostTracklet] = {}
        # base tracker id -> stitched id, while the stitched track is active
        self._alias: dict[int, int] = {}

    def track_frame(self, frame: np.ndarray, frame_idx: int) -> list[Track]:
        tracks = self.base.track_frame(frame, frame_idx)
        out: list[Track] = []
        seen: set[int] = set()

        for track in tracks:
            if track.track_id in self._alias:
                stitched_id = self._alias[track.track_id]
                if stitched_id not in seen:
                    track = replace(track, track_id=stitched_id)
            if track.track_id in self._active:
                self._append(self._active[track.track_id], track)
                seen.add(track.track_id)
                out.append(track)
                continue
            if track.track_id in self._lost:
                lost = self._lost.pop(track.track_id)
                hist = (lost.recent + [track])[-(self.velocity_window + 1) :]
                self._active[track.track_id] = hist
                seen.add(track.track_id)
                out.append(track)
                continue
            match_id = self._match(track, frame_idx, seen)
            if match_id is not None:
                lost = self._lost.pop(match_id)
                base_id = track.track_id
                track = replace(track, track_id=match_id)
                self._alias[base_id] = match_id
                self._active[match_id] = (lost.recent + [track])[-(self.velocity_window + 1) :]
                seen.add(match_id)
                logger.debug(
                    "stitched tracklet across %d-frame gap into id %d",
                    frame_idx - lost.last.frame_idx,
                    match_id,
                )
            else:
                self._active[track.track_id] = [track]
                seen.add(track.track_id)
            out.append(track)

        for tid in list(self._active):
            if tid not in seen:
                recent = self._active.pop(tid)
                self._lost[tid] = _LostTracklet(
                    last=recent[-1], velocity=_velocity(recent), recent=recent
                )
                self._alias = {b: m for b, m in self._alias.items() if m != tid}
        for tid in list(self._lost):
            if frame_idx - self._lost[tid].last.frame_idx > self.max_gap_frames:
                del self._lost[tid]
        return out

    def _append(self, hist: list[Track], track: Track) -> None:
        hist.append(track)
        del hist[: -(self.velocity_window + 1)]

    def _match(self, track: Track, frame_idx: int, seen: set[int]) -> int | None:
        fx, fy = _foot(track)
        best_id, best_d = None, self.max_dist_px
        for tid, lost in self._lost.items():
            if tid in seen:
                continue
            gap = frame_idx - lost.last.frame_idx
            if not 0 < gap <= self.max_gap_frames:
                continue
            lx, ly = _foot(lost.last)
            px = lx + lost.velocity[0] * gap
            py = ly + lost.velocity[1] * gap
            d = math.hypot(fx - px, fy - py)
            if d < best_d:
                best_id, best_d = tid, d
        return best_id
