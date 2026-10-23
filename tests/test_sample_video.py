import cv2
from scripts.make_sample_video import N_FRAMES, generate


def test_generate_sample_video(tmp_path):
    out = generate(tmp_path / "sample.mp4", seed=42)
    assert out.exists()

    cap = cv2.VideoCapture(str(out))
    frames = 0
    ok, frame = cap.read()
    assert ok
    height, width = frame.shape[:2]
    while ok:
        frames += 1
        ok, frame = cap.read()
    cap.release()

    assert frames == N_FRAMES
    assert (width, height) == (640, 360)


def test_generate_is_deterministic(tmp_path):
    a = generate(tmp_path / "a.mp4", seed=7)
    b = generate(tmp_path / "b.mp4", seed=7)
    cap_a, cap_b = cv2.VideoCapture(str(a)), cv2.VideoCapture(str(b))
    ok_a, frame_a = cap_a.read()
    ok_b, frame_b = cap_b.read()
    assert ok_a and ok_b
    assert (frame_a == frame_b).all()
    cap_a.release()
    cap_b.release()
