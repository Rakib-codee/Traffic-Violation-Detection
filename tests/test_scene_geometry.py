from tvd.config import Lane, Point, SceneConfig
from tvd.scene import (
    crosses_stop_line,
    direction_alignment,
    lane_for_point,
    point_in_polygon,
)

SQUARE = [Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10), Point(x=0, y=10)]


def _scene() -> SceneConfig:
    return SceneConfig(
        camera_id="test",
        stop_line=[Point(x=5, y=-5), Point(x=5, y=15)],
        lanes=[
            Lane(name="right", polygon=SQUARE, direction=Point(x=1, y=0)),
            Lane(
                name="left",
                polygon=[Point(x=20, y=0), Point(x=30, y=0), Point(x=30, y=10), Point(x=20, y=10)],
                direction=Point(x=1, y=0),
            ),
        ],
        signal_roi={"x": 0, "y": 0, "width": 10, "height": 20},
    )


def test_point_in_polygon():
    assert point_in_polygon(5, 5, SQUARE)
    assert not point_in_polygon(15, 5, SQUARE)
    assert not point_in_polygon(-1, 5, SQUARE)


def test_lane_for_point():
    scene = _scene()
    assert lane_for_point(scene, 5, 5).name == "right"
    assert lane_for_point(scene, 25, 5).name == "left"
    assert lane_for_point(scene, 100, 100) is None


def test_crosses_stop_line_vertical():
    scene = _scene()  # stop line x=5 from y=-5 to y=15
    assert crosses_stop_line(Point(x=0, y=5), Point(x=10, y=5), scene)
    assert not crosses_stop_line(Point(x=0, y=5), Point(x=3, y=5), scene)


def test_crosses_stop_line_horizontal():
    scene = SceneConfig(
        camera_id="h",
        stop_line=[Point(x=0, y=100), Point(x=200, y=100)],
        lanes=[Lane(name="l", polygon=SQUARE, direction=Point(x=0, y=1))],
        signal_roi={"x": 0, "y": 0, "width": 5, "height": 5},
    )
    assert crosses_stop_line(Point(x=50, y=90), Point(x=50, y=110), scene)
    assert not crosses_stop_line(Point(x=50, y=110), Point(x=50, y=120), scene)


def test_direction_alignment():
    lane = Lane(name="l", polygon=SQUARE, direction=Point(x=1, y=0))
    assert direction_alignment(Point(x=2, y=0), lane) > 0.99
    assert direction_alignment(Point(x=-2, y=0), lane) < -0.99
    assert abs(direction_alignment(Point(x=0, y=3), lane)) < 0.01
