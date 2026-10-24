"""Evidence clip builder.

For each violation event, extract a short clip around the event from the
source video, plus an annotated keyframe and JSON metadata. Blurring of
plates/faces is applied by default on stored outputs (privacy default).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from tvd.evidence.blur import blur_plate_band
from tvd.rules import ViolationEvent
from tvd.track.tracker import Track
from tvd.utils import get_logger

logger = get_logger(__name__)


@dataclass
class EvidenceItem:
    """One stored evidence package for a violation event."""

    event_id: str
    violation_type: str
    track_id: int
    camera_id: str
    start_frame: int
    end_frame: int
    clip_path: str
    frame_path: str
    metadata_path: str


class EvidenceBuilder:
    """Extracts clips + keyframes for events from the source video."""

    def __init__(
        self,
        video_path: Path,
        out_dir: Path,
        clip_seconds: float = 7.0,
        blur: bool = True,
        vehicle_boxes_by_frame: dict[int, list[tuple[int, int, int, int]]] | None = None,
    ):
        self.video_path = Path(video_path)
        self.out_dir = Path(out_dir)
        self.clip_seconds = clip_seconds
        self.blur = blur
        self.vehicle_boxes = vehicle_boxes_by_frame or {}

    def _window(self, event: ViolationEvent, fps: float, total_frames: int) -> tuple[int, int]:
        half = int(self.clip_seconds * fps / 2)
        start = max(event.start_frame - half, 1)
        end = min(event.end_frame + half, total_frames)
        return start, end

    def _maybe_blur(self, frame, frame_idx: int):
        if not self.blur:
            return frame
        return blur_plate_band(frame, self.vehicle_boxes.get(frame_idx, []))

    def build(
        self, event: ViolationEvent, keyframe_tracks: list[Track] | None = None
    ) -> EvidenceItem:
        """Write clip + keyframe + metadata for one event; returns the item."""
        event_id = (
            f"{event.camera_id}-{event.violation_type.value}-{event.track_id}-{event.start_frame}"
        )
        item_dir = self.out_dir / event_id
        item_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"cannot open video {self.video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start, end = self._window(event, fps, total)

        clip_path = item_dir / "clip.mp4"
        writer = cv2.VideoWriter(
            str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        keyframe_idx = (event.start_frame + event.end_frame) // 2
        frame_path = item_dir / "keyframe.jpg"

        cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        idx = start
        wrote_keyframe = False
        ok, frame = cap.read()
        while ok and idx <= end:
            frame = self._maybe_blur(frame, idx)
            writer.write(frame)
            if idx == keyframe_idx and not wrote_keyframe:
                for t in keyframe_tracks or []:
                    x1, y1, x2, y2 = int(t.x1), int(t.y1), int(t.x2), int(t.y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"id {t.track_id}",
                        (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                cv2.imwrite(str(frame_path), frame)
                wrote_keyframe = True
            idx += 1
            ok, frame = cap.read()
        writer.release()
        cap.release()

        if not wrote_keyframe:
            cv2.imwrite(str(frame_path), frame) if frame is not None else None

        metadata_path = item_dir / "metadata.json"
        metadata = {
            "event_id": event_id,
            "violation_type": event.violation_type.value,
            "track_id": event.track_id,
            "camera_id": event.camera_id,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "fps": fps,
            "clip_frames": [start, end],
            "blurred": self.blur,
            "note": "candidate event for human review; not an automated penalty",
            "details": event.details,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

        item = EvidenceItem(
            event_id=event_id,
            violation_type=event.violation_type.value,
            track_id=event.track_id,
            camera_id=event.camera_id,
            start_frame=event.start_frame,
            end_frame=event.end_frame,
            clip_path=str(clip_path),
            frame_path=str(frame_path),
            metadata_path=str(metadata_path),
        )
        logger.info("evidence built: %s (frames %d-%d)", event_id, start, end)
        return item


def item_to_dict(item: EvidenceItem) -> dict:
    return asdict(item)
