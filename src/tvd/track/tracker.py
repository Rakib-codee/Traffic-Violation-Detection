"""Multi-object tracker wrapper around ultralytics ByteTrack / BoT-SORT.

Thresholds come from TrackerConfig; the ultralytics tracker yaml is generated
from that config at runtime so there is a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from tvd.config import TrackerConfig
from tvd.utils import get_logger, set_seed

logger = get_logger(__name__)


@dataclass
class Track:
    """One tracked object in one frame (pixel coordinates)."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    frame_idx: int  # 1-based

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def box_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)


def write_tracker_yaml(config: TrackerConfig, out_path: Path) -> Path:
    """Render an ultralytics-compatible tracker yaml from TrackerConfig."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tracker_type": config.algorithm,  # bytetrack | botsort
        "track_high_thresh": config.track_high_thresh,
        "track_low_thresh": config.track_low_thresh,
        "new_track_thresh": config.track_high_thresh,
        "track_buffer": config.max_time_lost,
        "match_thresh": config.match_thresh,
        "fuse_score": True,
        # BoT-SORT extras (ignored by ByteTrack)
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "with_reid": False,
    }
    out_path.write_text(yaml.safe_dump(payload))
    return out_path


class Tracker:
    """Runs detector + tracker over frames and returns typed Track objects."""

    def __init__(
        self,
        config: TrackerConfig,
        weights: str,
        device: str = "auto",
        seed: int = 42,
        work_dir: Path = Path("results/tracking"),
    ):
        from tvd.detect.detector import pick_device

        self.config = config
        self.device = pick_device(device)
        set_seed(seed)
        self.tracker_yaml = write_tracker_yaml(config, Path(work_dir) / "tracker.yaml")

        from ultralytics import YOLO

        logger.info(
            "loading tracker weights=%s algorithm=%s device=%s",
            weights,
            config.algorithm,
            self.device,
        )
        self._model = YOLO(weights)

    def track_frame(self, frame: np.ndarray, frame_idx: int) -> list[Track]:
        """Advance tracking by one frame (1-based frame_idx)."""
        results = self._model.track(
            frame,
            persist=True,
            tracker=str(self.tracker_yaml),
            conf=self.config.track_low_thresh,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        tracks: list[Track] = []
        if result.boxes.id is None:
            return tracks
        names = result.names
        for box in result.boxes:
            track_id = int(box.id.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            tracks.append(
                Track(
                    track_id=track_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf.item()),
                    class_name=names[int(box.cls.item())],
                    frame_idx=frame_idx,
                )
            )
        return tracks
