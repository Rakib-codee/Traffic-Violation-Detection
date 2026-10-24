"""Augmentation transform tests."""

import numpy as np

from tvd.detect.augment import TRANSFORMS, augment_frame


def _frame(seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


def test_all_transforms_preserve_shape_and_dtype():
    frame = _frame()
    for kind in TRANSFORMS:
        out = augment_frame(frame, kind, seed=42)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8


def test_transforms_are_deterministic():
    frame = _frame()
    for kind in TRANSFORMS:
        a = augment_frame(frame, kind, seed=7)
        b = augment_frame(frame, kind, seed=7)
        assert np.array_equal(a, b)


def test_night_darkens():
    frame = _frame()
    out = augment_frame(frame, "night", seed=1)
    assert out.mean() < frame.mean()


def test_transforms_change_pixels():
    frame = _frame()
    for kind in TRANSFORMS:
        out = augment_frame(frame, kind, seed=1)
        assert not np.array_equal(out, frame), f"{kind} was a no-op"
