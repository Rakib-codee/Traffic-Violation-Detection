"""End-to-end smoke: run_video over the synthetic sample clip."""

import cv2
from scripts.make_sample_video import N_FRAMES, generate


def test_run_video_outputs(tmp_path):
    from tvd.config import load_tracker_config
    from tvd.track.pipeline import track_video
    from tvd.track.tracker import Tracker

    video = generate(tmp_path / "sample.mp4", seed=42)
    cfg = load_tracker_config("configs/tracker.yaml")
    tracker = Tracker(cfg, weights="yolov8n.pt", device="auto", work_dir=tmp_path / "work")
    summary = track_video(video, tracker, tmp_path / "out", cfg, "yolov8n.pt", annotate=True)

    out = tmp_path / "out"
    assert (out / "tracks.txt").exists()
    assert (out / "trajectories.csv").exists()
    assert (out / "summary.json").exists()
    assert summary["frames"] == N_FRAMES

    cap = cv2.VideoCapture(str(out / "annotated.mp4"))
    frames = 0
    ok, _ = cap.read()
    while ok:
        frames += 1
        ok, _ = cap.read()
    cap.release()
    assert frames == N_FRAMES
