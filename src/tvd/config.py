"""Typed configuration schema and YAML loaders.

All pipeline behaviour is driven by YAML configs validated here;
no magic numbers in component code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class SystemConfig(BaseModel):
    """Top-level run configuration (configs/system.yaml)."""

    seed: int = Field(default=42, description="Global RNG seed, logged per run.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    sample_video: Path = Path("data/samples/synthetic_intersection.mp4")


class DetectorConfig(BaseModel):
    """Object detector configuration (configs/model.yaml)."""

    architecture: Literal["yolov8", "yolov11", "rt-detr"] = "yolov8"
    weights: Path = Path("weights/detector.pt")
    conf_threshold: float = Field(default=0.25, gt=0.0, lt=1.0)
    iou_threshold: float = Field(default=0.45, gt=0.0, lt=1.0)
    input_size: int = Field(default=640, gt=0)
    classes: list[str] = Field(default=["car", "bus", "truck", "motorcycle", "person"])


class TrackerConfig(BaseModel):
    """Multi-object tracker configuration (configs/tracker.yaml)."""

    algorithm: Literal["bytetrack", "bot-sort"] = "bytetrack"
    track_high_thresh: float = Field(default=0.5, gt=0.0, lt=1.0)
    track_low_thresh: float = Field(default=0.1, gt=0.0, lt=1.0)
    match_thresh: float = Field(default=0.8, gt=0.0, lt=1.0)
    max_time_lost: int = Field(
        default=30, gt=0, description="Frames before a lost track is dropped."
    )
    min_track_duration: int = Field(
        default=5, gt=0, description="Minimum frames for a track to feed the rule engine."
    )

    @field_validator("track_low_thresh")
    @classmethod
    def low_below_high(cls, v: float, info) -> float:
        high = info.data.get("track_high_thresh")
        if high is not None and v >= high:
            raise ValueError("track_low_thresh must be below track_high_thresh")
        return v


class Point(BaseModel):
    x: float
    y: float


class Lane(BaseModel):
    """A lane polygon with its expected travel direction."""

    name: str
    polygon: list[Point]
    direction: Point = Field(description="Unit-ish vector of expected travel direction.")

    @field_validator("polygon")
    @classmethod
    def at_least_three_points(cls, v: list[Point]) -> list[Point]:
        if len(v) < 3:
            raise ValueError("lane polygon needs at least 3 points")
        return v


class SignalRoi(BaseModel):
    """Region of interest containing the traffic signal."""

    x: float
    y: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class SceneConfig(BaseModel):
    """Per-camera scene calibration (configs/scene.<camera>.yaml)."""

    camera_id: str
    stop_line: list[Point] = Field(min_length=2, max_length=2)
    lanes: list[Lane] = Field(min_length=1)
    signal_roi: SignalRoi
    red_light_tolerance_s: float = Field(
        default=0.3, ge=0.0, description="Grace window after signal turns red."
    )
    wrong_way_min_frames: int = Field(
        default=10, gt=0, description="Frames of opposing motion before flagging."
    )

    @field_validator("lanes")
    @classmethod
    def unique_lane_names(cls, v: list[Lane]) -> list[Lane]:
        names = [lane.name for lane in v]
        if len(names) != len(set(names)):
            raise ValueError("lane names must be unique")
        return v


def load_yaml(path: Path | str) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_system_config(path: Path | str) -> SystemConfig:
    return SystemConfig(**load_yaml(path))


def load_detector_config(path: Path | str) -> DetectorConfig:
    return DetectorConfig(**load_yaml(path))


def load_tracker_config(path: Path | str) -> TrackerConfig:
    return TrackerConfig(**load_yaml(path))


def load_scene_config(path: Path | str) -> SceneConfig:
    return SceneConfig(**load_yaml(path))
