import pytest
from scripts.annotate_scene import AnnotationCollector

from tvd.config import load_scene_config


def test_collector_builds_valid_scene(tmp_path):
    c = AnnotationCollector()
    c.add_stop_line_point(320, 400)
    c.add_stop_line_point(320, 600)

    c.add_lane_point(340, 400)
    c.add_lane_point(480, 400)
    c.add_lane_point(480, 720)
    c.add_lane_point(340, 720)
    c.finish_lane_polygon()
    c.set_lane_direction(410, 560, 410, 700)

    c.set_signal_roi(300, 120, 40, 90)
    scene = c.to_scene_config("cam-test")
    assert scene.camera_id == "cam-test"
    assert len(scene.lanes) == 1
    assert scene.lanes[0].name == "lane-1"
    assert scene.signal_roi.width == 40


def test_collector_requires_two_stop_line_points():
    c = AnnotationCollector()
    c.add_stop_line_point(1, 2)
    with pytest.raises(ValueError, match="stop line"):
        c.to_scene_config("x")


def test_collector_rejects_third_stop_line_point():
    c = AnnotationCollector()
    c.add_stop_line_point(1, 2)
    c.add_stop_line_point(3, 4)
    with pytest.raises(ValueError, match="2 points"):
        c.add_stop_line_point(5, 6)


def test_collector_requires_direction_arrow():
    c = AnnotationCollector()
    c.add_stop_line_point(0, 0)
    c.add_stop_line_point(10, 0)
    c.add_lane_point(0, 0)
    c.add_lane_point(10, 0)
    c.add_lane_point(10, 10)
    c.finish_lane_polygon()
    c.set_signal_roi(0, 0, 5, 5)
    with pytest.raises(ValueError, match="direction"):
        c.to_scene_config("x")


def test_collector_requires_three_polygon_points():
    c = AnnotationCollector()
    c.add_lane_point(0, 0)
    c.add_lane_point(10, 0)
    with pytest.raises(ValueError, match="3 points"):
        c.finish_lane_polygon()


def test_scene_config_written_file_loads(tmp_path):
    c = AnnotationCollector()
    c.add_stop_line_point(5, -5)
    c.add_stop_line_point(5, 15)
    c.add_lane_point(0, 0)
    c.add_lane_point(10, 0)
    c.add_lane_point(10, 10)
    c.add_lane_point(0, 10)
    c.finish_lane_polygon()
    c.set_lane_direction(5, 5, 8, 5)
    c.set_signal_roi(20, 20, 10, 30)
    scene = c.to_scene_config("roundtrip")
    path = tmp_path / "scene.roundtrip.yaml"
    path.write_text(scene.model_dump_json())
    loaded = load_scene_config(path)
    assert loaded.lanes[0].name == scene.lanes[0].name
