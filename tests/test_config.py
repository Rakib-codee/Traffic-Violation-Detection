from pathlib import Path

import pytest

from tvd.config import (
    DetectorConfig,
    SceneConfig,
    SystemConfig,
    TrackerConfig,
    load_detector_config,
    load_scene_config,
    load_system_config,
    load_tracker_config,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_system_config_loads():
    cfg = load_system_config(CONFIGS / "system.yaml")
    assert isinstance(cfg, SystemConfig)
    assert cfg.seed == 42


def test_detector_config_loads():
    cfg = load_detector_config(CONFIGS / "model.yaml")
    assert isinstance(cfg, DetectorConfig)
    assert "car" in cfg.classes


def test_tracker_config_loads():
    cfg = load_tracker_config(CONFIGS / "tracker.yaml")
    assert isinstance(cfg, TrackerConfig)
    assert cfg.algorithm == "bytetrack"


def test_scene_config_loads():
    cfg = load_scene_config(CONFIGS / "scene.example.yaml")
    assert isinstance(cfg, SceneConfig)
    assert len(cfg.stop_line) == 2
    assert {lane.name for lane in cfg.lanes} == {"northbound", "southbound"}


def test_tracker_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        TrackerConfig(track_high_thresh=0.2, track_low_thresh=0.5)


def test_scene_rejects_degenerate_lane(tmp_path):
    bad = tmp_path / "bad_scene.yaml"
    bad.write_text(
        "camera_id: c1\n"
        "stop_line: [{x: 0, y: 0}, {x: 1, y: 1}]\n"
        "lanes:\n"
        "  - name: only-two-points\n"
        "    polygon: [{x: 0, y: 0}, {x: 1, y: 1}]\n"
        "    direction: {x: 0, y: 1}\n"
        "signal_roi: {x: 0, y: 0, width: 10, height: 10}\n"
    )
    with pytest.raises(ValueError):
        load_scene_config(bad)


def test_scene_rejects_duplicate_lane_names():
    from pydantic import ValidationError

    lane = {
        "name": "dup",
        "polygon": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
        "direction": {"x": 0, "y": 1},
    }
    with pytest.raises(ValidationError):
        SceneConfig(
            camera_id="c1",
            stop_line=[{"x": 0, "y": 0}, {"x": 1, "y": 1}],
            lanes=[lane, lane],
            signal_roi={"x": 0, "y": 0, "width": 5, "height": 5},
        )
