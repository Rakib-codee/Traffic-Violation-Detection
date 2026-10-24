"""Phase 3 exit criterion: every committed scene config must validate."""

from pathlib import Path

from tvd.config import load_scene_config

SCENE_CONFIGS = sorted(Path("configs").glob("scene.*.yaml"))


def test_at_least_two_camera_views_committed():
    assert len(SCENE_CONFIGS) >= 2, "need scene configs for 2+ camera views"


def test_all_committed_scene_configs_validate():
    assert SCENE_CONFIGS, "no scene configs found"
    for path in SCENE_CONFIGS:
        scene = load_scene_config(path)
        assert scene.lanes, f"{path}: no lanes"
        assert len(scene.stop_line) == 2, f"{path}: stop line must have 2 points"
