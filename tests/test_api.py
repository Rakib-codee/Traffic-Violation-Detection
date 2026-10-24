from fastapi.testclient import TestClient
from scripts.make_sample_video import generate

from tvd.api.main import create_app
from tvd.evidence import EventStore, EvidenceBuilder
from tvd.rules import ViolationEvent, ViolationType


def _seeded_app(tmp_path):
    store = EventStore(tmp_path / "events.db")
    video = generate(tmp_path / "v.mp4", seed=42)
    event = ViolationEvent(
        violation_type=ViolationType.WRONG_WAY,
        track_id=9,
        start_frame=5,
        end_frame=15,
        camera_id="cam-api",
        details={},
    )
    item = EvidenceBuilder(video, tmp_path / "ev").build(event)
    store.add_event(event, item)
    return TestClient(create_app(str(tmp_path / "events.db"))), item.event_id


def test_health(tmp_path):
    client, _ = _seeded_app(tmp_path)
    assert client.get("/health").json() == {"status": "ok"}


def test_list_and_get_event(tmp_path):
    client, event_id = _seeded_app(tmp_path)
    events = client.get("/events").json()
    assert len(events) == 1
    assert events[0]["violation_type"] == "wrong_way"

    detail = client.get(f"/events/{event_id}").json()
    assert detail["track_id"] == 9

    filtered = client.get("/events", params={"violation_type": "red_light"}).json()
    assert filtered == []


def test_missing_event_404(tmp_path):
    client, _ = _seeded_app(tmp_path)
    assert client.get("/events/missing").status_code == 404


def test_clip_and_keyframe_endpoints(tmp_path):
    client, event_id = _seeded_app(tmp_path)
    clip = client.get(f"/events/{event_id}/clip")
    assert clip.status_code == 200
    assert clip.headers["content-type"] == "video/mp4"
    keyframe = client.get(f"/events/{event_id}/keyframe")
    assert keyframe.status_code == 200
    assert keyframe.headers["content-type"] == "image/jpeg"


def test_metadata_endpoint(tmp_path):
    client, event_id = _seeded_app(tmp_path)
    meta = client.get(f"/events/{event_id}/metadata").json()
    assert meta["violation_type"] == "wrong_way"
    assert "human review" in meta["note"]
