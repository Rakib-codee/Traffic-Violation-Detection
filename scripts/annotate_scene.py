"""Interactive scene calibration tool.

Draw the stop line, lane polygons with direction arrows, and the signal ROI
on one frame of a camera view; writes a validated scene.<camera>.yaml.

Usage:
    python scripts/annotate_scene.py --camera-id cam-01 \
        --video data/samples/synthetic_intersection.mp4 [--frame-idx 0] \
        --out configs/scene.cam-01.yaml

Or from a still image: --image path.jpg

Controls: left-click = add point / drag rectangle (signal ROI step)
          n = finish current step (lane polygon / direction arrow)
          u = undo last point
          q = finish lane entry (all lanes done)
The headless geometry logic lives in AnnotationCollector and is unit-tested.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from tvd.config import Lane, Point, SceneConfig, SignalRoi, load_scene_config
from tvd.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LaneAnnotation:
    name: str
    polygon: list[tuple[float, float]] = field(default_factory=list)
    direction: tuple[tuple[float, float], tuple[float, float]] | None = None


class AnnotationCollector:
    """Collects scene geometry without any GUI (testable core)."""

    def __init__(self):
        self.stop_line: list[tuple[float, float]] = []
        self.lanes: list[LaneAnnotation] = []
        self.signal_roi: tuple[float, float, float, float] | None = None

    def add_stop_line_point(self, x: float, y: float) -> None:
        if len(self.stop_line) >= 2:
            raise ValueError("stop line already has 2 points")
        self.stop_line.append((x, y))

    def add_lane_point(self, x: float, y: float) -> None:
        if not self.lanes:
            self.lanes.append(LaneAnnotation(name=""))
        self.lanes[-1].polygon.append((x, y))

    def finish_lane_polygon(self) -> None:
        lane = self.lanes[-1]
        if len(lane.polygon) < 3:
            raise ValueError(f"lane polygon needs >= 3 points, got {len(lane.polygon)}")
        if lane.name == "":
            lane.name = f"lane-{len(self.lanes)}"

    def set_lane_direction(self, x_from: float, y_from: float, x_to: float, y_to: float) -> None:
        lane = self.lanes[-1]
        if len(lane.polygon) < 3:
            raise ValueError("finish the lane polygon before its direction arrow")
        lane.direction = ((x_from, y_from), (x_to, y_to))

    def set_signal_roi(self, x: float, y: float, w: float, h: float) -> None:
        if w <= 0 or h <= 0:
            raise ValueError("signal ROI must have positive width and height")
        self.signal_roi = (x, y, w, h)

    def to_scene_config(self, camera_id: str) -> SceneConfig:
        if len(self.stop_line) != 2:
            raise ValueError("stop line needs exactly 2 points")
        if not self.lanes:
            raise ValueError("at least one lane is required")

        lanes: list[Lane] = []
        for ann in self.lanes:
            if ann.direction is None:
                raise ValueError(f"lane '{ann.name}' is missing its direction arrow")
            (x1, y1), (x2, y2) = ann.direction
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                raise ValueError(f"lane '{ann.name}' has a zero direction vector")
            lanes.append(
                Lane(
                    name=ann.name,
                    polygon=[Point(x=x, y=y) for x, y in ann.polygon],
                    direction=Point(x=dx, y=dy),
                )
            )

        signal_roi = None
        if self.signal_roi is not None:
            x, y, w, h = self.signal_roi
            signal_roi = SignalRoi(x=x, y=y, width=w, height=h)
        return SceneConfig(
            camera_id=camera_id,
            stop_line=[Point(x=px, y=py) for px, py in self.stop_line],
            lanes=lanes,
            signal_roi=signal_roi,
        )


def load_frame(image: Path | None, video: Path | None, frame_idx: int) -> tuple:
    if image is not None:
        frame = cv2.imread(str(image))
        if frame is None:
            raise SystemExit(f"cannot read image {image}")
        return frame
    if video is None:
        raise SystemExit("provide --image or --video")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {frame_idx} from {video}")
    return frame


def run_interactive(collector: AnnotationCollector, frame, camera_id: str, out_path: Path) -> None:
    """cv2 mouse-driven UI; thin wrapper over AnnotationCollector."""
    window = "annotate_scene"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    state = {"mode": "stop_line", "drag_start": None, "current": (0, 0)}

    def redraw():
        canvas = frame.copy()
        if len(collector.stop_line) == 2:
            cv2.line(canvas, collector.stop_line[0], collector.stop_line[1], (0, 0, 255), 2)
        for lane in collector.lanes:
            if len(lane.polygon) > 1:
                pts = lane.polygon + [state["current"]] if state["mode"] == "lane" else lane.polygon
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 200, 0), 2)
            if lane.direction:
                cv2.arrowedLine(canvas, lane.direction[0], lane.direction[1], (255, 200, 0), 2)
        if state["mode"] == "roi" and state["drag_start"]:
            x1, y1 = state["drag_start"]
            x2, y2 = state["current"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imshow(window, canvas)

    def on_mouse(event, x, y, _flags, _param):
        state["current"] = (x, y)
        if state["mode"] == "roi":
            if event == cv2.EVENT_LBUTTONDOWN:
                state["drag_start"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and state["drag_start"]:
                x1, y1 = state["drag_start"]
                collector.set_signal_roi(min(x1, x), min(y1, y), abs(x - x1), abs(y - y1))
                state["mode"] = "done"
                state["drag_start"] = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            if state["mode"] == "stop_line":
                collector.add_stop_line_point(x, y)
                if len(collector.stop_line) == 2:
                    state["mode"] = "lane"
            elif state["mode"] == "lane":
                collector.add_lane_point(x, y)
            elif state["mode"] == "direction" and state.get("arrow_start"):
                collector.set_lane_direction(*state["arrow_start"], x, y)
                state["mode"] = "lane-new"
        redraw()

    cv2.setMouseCallback(window, on_mouse)
    print(
        "Controls: click points; 'n' finish polygon then click the direction tip; "
        "'q' all lanes done; 'u' undo; Esc quit"
    )

    while True:
        redraw()
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # Esc
            cv2.destroyAllWindows()
            raise SystemExit("aborted")
        if key == ord("u"):
            if state["mode"] == "stop_line":
                target = collector.stop_line
            elif collector.lanes:
                target = collector.lanes[-1].polygon
            else:
                target = None
            if target:
                target.pop()
        if key == ord("n"):
            if state["mode"] == "lane":
                collector.finish_lane_polygon()
                # anchor the direction arrow at the polygon centroid
                poly = collector.lanes[-1].polygon
                state["arrow_start"] = (
                    sum(p[0] for p in poly) // len(poly),
                    sum(p[1] for p in poly) // len(poly),
                )
                state["mode"] = "direction"
                print("now click the direction arrow tip for this lane")
            elif state["mode"] == "lane-new":
                default = f"lane-{len(collector.lanes) + 1}"
                name = input(f"next lane name (empty = {default}): ").strip()
                collector.lanes.append(LaneAnnotation(name=name or default))
                state["mode"] = "lane"
        if key == ord("q") and state["mode"] in ("lane", "lane-new", "direction"):
            last = collector.lanes[-1] if collector.lanes else None
            unfinished = last and len(last.polygon) >= 3 and last.direction is None
            if state["mode"] == "lane" and unfinished:
                print("finish the current lane first (n, then direction arrow)")
                continue
            state["mode"] = "roi"
            print("drag the signal ROI rectangle")
        if state["mode"] == "done":
            break
    cv2.destroyAllWindows()

    scene = collector.to_scene_config(camera_id)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(json.loads(scene.model_dump_json()), sort_keys=False))
    load_scene_config(out_path)  # validate what we wrote
    logger.info("wrote and validated %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = load_frame(args.image, args.video, args.frame_idx)
    run_interactive(AnnotationCollector(), frame, args.camera_id, args.out)


if __name__ == "__main__":
    main()
