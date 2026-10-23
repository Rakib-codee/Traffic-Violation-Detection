"""MOTChallenge-format track I/O.

Detection/result lines: frame, id, x, y, w, h, conf, -1, -1, -1
GT lines add class and visibility instead of the three -1s.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tvd.track.tracker import Track


@dataclass
class TrackRow:
    """One MOT-format row."""

    frame: int  # 1-based
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0
    cls: int = -1
    visibility: float = 1.0

    def to_line(self) -> str:
        return (
            f"{self.frame},{self.track_id},{self.x:.2f},{self.y:.2f},"
            f"{self.w:.2f},{self.h:.2f},{self.conf:.4f},{self.cls},{self.visibility:.4f}"
        )


def track_to_row(track: Track) -> TrackRow:
    x, y, w, h = track.box_xywh()
    return TrackRow(
        frame=track.frame_idx,
        track_id=track.track_id,
        x=x,
        y=y,
        w=w,
        h=h,
        conf=track.confidence,
    )


def write_mot(tracks: list[TrackRow], out_path: Path) -> Path:
    """Write MOT-format rows sorted by (frame, id); creates parent dirs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [row.to_line() for row in sorted(tracks, key=lambda r: (r.frame, r.track_id))]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return out_path


def read_mot(path: Path) -> list[TrackRow]:
    rows: list[TrackRow] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise ValueError(f"{path}: malformed MOT line: {line!r}")
        rows.append(
            TrackRow(
                frame=int(parts[0]),
                track_id=int(parts[1]),
                x=float(parts[2]),
                y=float(parts[3]),
                w=float(parts[4]),
                h=float(parts[5]),
                conf=float(parts[6]) if len(parts) > 6 else 1.0,
                cls=int(float(parts[7])) if len(parts) > 7 else -1,
            )
        )
    return rows
