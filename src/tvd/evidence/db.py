"""SQLite event store for violation events and evidence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tvd.evidence.builder import EvidenceItem
from tvd.rules import ViolationEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    violation_type TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    camera_id TEXT NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    clip_path TEXT NOT NULL,
    frame_path TEXT NOT NULL,
    metadata_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class EventStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_event(self, event: ViolationEvent, item: EvidenceItem) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO events "
                "(event_id, violation_type, track_id, camera_id, start_frame, end_frame, "
                " clip_path, frame_path, metadata_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.event_id,
                    event.violation_type.value,
                    event.track_id,
                    event.camera_id,
                    event.start_frame,
                    event.end_frame,
                    item.clip_path,
                    item.frame_path,
                    item.metadata_path,
                ),
            )

    def list_events(
        self, violation_type: str | None = None, camera_id: str | None = None
    ) -> list[dict]:
        query = "SELECT * FROM events WHERE 1=1"
        params: list = []
        if violation_type:
            query += " AND violation_type = ?"
            params.append(violation_type)
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        query += " ORDER BY created_at DESC"
        with self._conn() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def get_event(self, event_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
        return dict(row) if row else None
