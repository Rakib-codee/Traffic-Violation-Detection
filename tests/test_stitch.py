import numpy as np

from tvd.track.stitch import TrackStitcher, _velocity
from tvd.track.tracker import Track


class _ScriptedTracker:
    """Fake base tracker returning scripted tracks per frame."""

    def __init__(self, script):
        self._script = script  # frame_idx -> list[Track]

    def track_frame(self, frame, frame_idx):
        return self._script.get(frame_idx, [])


def track(tid, x, y_foot, frame, size=20.0):
    return Track(
        track_id=tid,
        x1=x - size / 2,
        y1=y_foot - size * 2,
        x2=x + size / 2,
        y2=y_foot,
        confidence=0.9,
        class_name="car",
        frame_idx=frame,
    )


def moving(tid, frames, x=100.0, y0=100.0, vy=5.0):
    """Script fragment: track moving vy px/frame over the given frames."""
    return {f: [track(tid, x, y0 + vy * (f - frames[0]), f)] for f in frames}


def run(stitcher, n_frames):
    out = {}
    for f in range(1, n_frames + 1):
        out[f] = stitcher.track_frame(np.zeros((10, 10, 3), dtype=np.uint8), f)
    return out


def merge_scripts(*scripts):
    merged = {}
    for script in scripts:
        for f, tracks in script.items():
            merged.setdefault(f, []).extend(tracks)
    return merged


def test_stitches_across_short_gap():
    script = merge_scripts(moving(1, range(1, 6)), moving(2, range(9, 12), y0=140.0))
    out = run(TrackStitcher(_ScriptedTracker(script), max_gap_frames=10, max_dist_px=80), 11)
    assert [t.track_id for t in out[9]] == [1]
    assert [t.track_id for t in out[10]] == [1]


def test_no_stitch_beyond_gap_window():
    script = merge_scripts(moving(1, range(1, 6)), moving(2, range(20, 23), y0=140.0))
    out = run(TrackStitcher(_ScriptedTracker(script), max_gap_frames=10, max_dist_px=80), 22)
    assert [t.track_id for t in out[20]] == [2]


def test_no_stitch_beyond_distance_gate():
    script = merge_scripts(moving(1, range(1, 6)), moving(2, range(7, 10), x=400.0, y0=400.0))
    out = run(TrackStitcher(_ScriptedTracker(script), max_gap_frames=10, max_dist_px=80), 9)
    assert [t.track_id for t in out[7]] == [2]


def test_match_uses_velocity_prediction_not_last_position():
    # v=(0,10)/frame, gap=5 -> prediction is 50 px ahead of the last position;
    # with a 40 px gate only the velocity-corrected match passes
    script = merge_scripts(moving(1, range(1, 6), vy=10.0), moving(2, range(10, 12), y0=190.0))
    # last seen at frame 5, foot y=140; gap 5 -> predicted y=190
    out = run(TrackStitcher(_ScriptedTracker(script), max_gap_frames=10, max_dist_px=40), 11)
    assert [t.track_id for t in out[10]] == [1]


def test_stationary_tracklet_stitches_at_last_position():
    script = merge_scripts(moving(1, range(1, 6), vy=0.0), moving(2, range(8, 10), vy=0.0))
    out = run(TrackStitcher(_ScriptedTracker(script), max_gap_frames=10, max_dist_px=10), 9)
    assert [t.track_id for t in out[8]] == [1]


def test_nearest_lost_tracklet_wins():
    script = merge_scripts(
        moving(1, range(1, 6), x=100.0),
        moving(2, range(1, 6), x=200.0),
        moving(3, range(7, 9), x=190.0),
    )
    out = run(TrackStitcher(_ScriptedTracker(script), max_gap_frames=10, max_dist_px=80), 8)
    assert [t.track_id for t in out[7]] == [2]


def test_velocity_uses_consecutive_pairs_only():
    tracks = [track(1, 100, 100, 1), track(1, 100, 110, 2), track(1, 100, 200, 10)]
    assert _velocity(tracks) == (0.0, 10.0)
