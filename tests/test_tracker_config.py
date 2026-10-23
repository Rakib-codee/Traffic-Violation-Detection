import yaml

from tvd.config import TrackerConfig, load_tracker_config
from tvd.track.tracker import write_tracker_yaml


def test_write_tracker_yaml_roundtrip(tmp_path):
    cfg = TrackerConfig(track_high_thresh=0.6, track_low_thresh=0.05, match_thresh=0.7)
    path = write_tracker_yaml(cfg, tmp_path / "tracker.yaml")
    data = yaml.safe_load(path.read_text())
    assert data["tracker_type"] == "bytetrack"
    assert data["track_high_thresh"] == 0.6
    assert data["track_low_thresh"] == 0.05
    assert data["new_track_thresh"] == 0.6
    assert data["match_thresh"] == 0.7
    assert data["track_buffer"] == cfg.max_time_lost


def test_committed_tracker_config_loads():
    cfg = load_tracker_config("configs/tracker.yaml")
    assert cfg.algorithm == "bytetrack"
    assert cfg.track_low_thresh < cfg.track_high_thresh
