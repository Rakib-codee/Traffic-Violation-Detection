"""Deterministic seeding helpers."""

from __future__ import annotations

import random

import numpy as np

from tvd.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> int:
    """Seed python and numpy RNGs; returns the seed so callers can log it."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    logger.info("seed set to %d", seed)
    return seed
