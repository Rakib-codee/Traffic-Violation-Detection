import pytest

from tvd.track.mot import TrackRow, read_mot, write_mot


def test_mot_roundtrip(tmp_path):
    rows = [
        TrackRow(frame=1, track_id=7, x=10.0, y=20.0, w=30.0, h=40.0, conf=0.9),
        TrackRow(frame=1, track_id=3, x=1.0, y=2.0, w=3.0, h=4.0),
        TrackRow(frame=2, track_id=7, x=11.0, y=20.0, w=30.0, h=40.0),
    ]
    out = write_mot(rows, tmp_path / "tracks.txt")
    loaded = read_mot(out)
    assert [r.to_line() for r in loaded] == [
        rows[1].to_line(),  # sorted by (frame, id): id 3 before id 7
        rows[0].to_line(),
        rows[2].to_line(),
    ]


def test_read_mot_rejects_bad_line(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("1,2,3\n")
    with pytest.raises(ValueError, match="malformed"):
        read_mot(bad)
