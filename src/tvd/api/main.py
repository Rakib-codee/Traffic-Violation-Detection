"""FastAPI backend exposing violation events and evidence.

Every event is a candidate for human review; this API serves reviewers.
Run: uvicorn tvd.api.main:app --port 8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from tvd.evidence.db import EventStore

DEFAULT_DB = "results/events.db"


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(title="Traffic Violation Detection — Review API")
    store = EventStore(db_path or os.environ.get("TVD_DB", DEFAULT_DB))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/events")
    def list_events(violation_type: str | None = None, camera_id: str | None = None) -> list[dict]:
        return store.list_events(violation_type=violation_type, camera_id=camera_id)

    @app.get("/events/{event_id}")
    def get_event(event_id: str) -> dict:
        event = store.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        return event

    @app.get("/events/{event_id}/metadata")
    def get_event_metadata(event_id: str) -> dict:
        event = store.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        path = Path(event["metadata_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="metadata file missing")
        return json.loads(path.read_text())

    @app.get("/events/{event_id}/clip")
    def get_event_clip(event_id: str) -> FileResponse:
        event = store.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        path = Path(event["clip_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="clip file missing")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/events/{event_id}/keyframe")
    def get_event_keyframe(event_id: str) -> FileResponse:
        event = store.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        path = Path(event["frame_path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail="keyframe file missing")
        return FileResponse(path, media_type="image/jpeg", filename=path.name)

    return app


app = create_app()
