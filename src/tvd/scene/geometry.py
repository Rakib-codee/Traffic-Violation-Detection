"""Scene geometry helpers: polygons, lanes, stop-line crossing.

Pure functions on the typed SceneConfig from tvd.config so the rule engine
(Phase 4) and the annotation tool both share one source of truth.
"""

from __future__ import annotations

from tvd.config import Lane, Point, SceneConfig


def point_in_polygon(x: float, y: float, polygon: list[Point]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        intersects = (a.y > y) != (b.y > y) and x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x
        if intersects:
            inside = not inside
    return inside


def lane_for_point(scene: SceneConfig, x: float, y: float) -> Lane | None:
    """Return the lane whose polygon contains the point, else None."""
    for lane in scene.lanes:
        if point_in_polygon(x, y, lane.polygon):
            return lane
    return None


def crosses_stop_line(prev: Point, curr: Point, scene: SceneConfig) -> bool:
    """True if the segment prev->curr intersects the stop line segment.

    Uses orientation tests so it works for any line orientation, not just
    horizontal stop lines.
    """
    a, b = scene.stop_line[0], scene.stop_line[1]

    def orient(p: Point, q: Point, r: Point) -> float:
        return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)

    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return min(p.x, r.x) <= q.x <= max(p.x, r.x) and min(p.y, r.y) <= q.y <= max(p.y, r.y)

    d1 = orient(a, b, prev)
    d2 = orient(a, b, curr)
    d3 = orient(prev, curr, a)
    d4 = orient(prev, curr, b)

    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    if d1 == 0 and on_segment(a, prev, b):
        return True
    if d2 == 0 and on_segment(a, curr, b):
        return True
    if d3 == 0 and on_segment(prev, a, curr):
        return True
    if d4 == 0 and on_segment(prev, b, curr):
        return True
    return False


def direction_alignment(motion: Point, lane: Lane) -> float:
    """Cosine similarity between a motion vector and the lane's expected direction.

    +1 = moving with the lane, -1 = directly opposing (wrong-way).
    """
    import math

    dot = motion.x * lane.direction.x + motion.y * lane.direction.y
    norm = math.hypot(motion.x, motion.y) * math.hypot(lane.direction.x, lane.direction.y)
    if norm == 0:
        return 0.0
    return dot / norm
