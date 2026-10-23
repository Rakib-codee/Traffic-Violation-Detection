"""Synthetic TrackEval benchmark: perfect predictions must score ~1.0,
an ID switch must degrade IDF1. Verifies the HOTA/IDF1 wiring end to end."""

import configparser
from pathlib import Path

import pytest

from tvd.track.evaluate import run_trackeval

N_FRAMES = 40
CLASS_VEHICLE = 1  # trackeval 'pedestrian' slot; stands for vehicles here


def _gt_track(track_id: int, start: int, end: int, x0: int, y0: int):
    """Straight-line GT track, one MOT row per frame."""
    rows = []
    for f in range(start, end + 1):
        x = x0 + 2 * (f - start)
        rows.append(f"{f},{track_id},{x},{y0},30,20,1,{CLASS_VEHICLE},1.0")
    return rows


def _make_benchmark(root: Path) -> tuple[Path, Path]:
    gt_dir = root / "gt"
    trackers_dir = root / "trackers"

    seq_dir = gt_dir / "testseq1"
    (seq_dir / "gt").mkdir(parents=True)
    gt_rows = _gt_track(1, 1, N_FRAMES, 50, 100) + _gt_track(2, 5, N_FRAMES, 200, 300)
    (seq_dir / "gt" / "gt.txt").write_text("\n".join(gt_rows) + "\n")

    ini = configparser.ConfigParser()
    ini["Sequence"] = {
        "name": "testseq1",
        "imDir": "img1",
        "frameRate": 25,
        "seqLength": N_FRAMES,
        "imWidth": 640,
        "imHeight": 480,
        "imExt": ".jpg",
    }
    with (seq_dir / "seqinfo.ini").open("w") as f:
        ini.write(f)

    # tracker "perfect": GT boxes verbatim (conf column replaced, cls=-1)
    for tracker_name, switch_ids in (("perfect", False), ("idswitch", True)):
        data_dir = trackers_dir / tracker_name / "data"
        data_dir.mkdir(parents=True)
        pred_rows = []
        for row in gt_rows:
            parts = row.split(",")
            frame, tid, x, y, w, h = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            if switch_ids and int(tid) == 1 and int(frame) > N_FRAMES // 2:
                tid = "9"  # id 1 becomes id 9 halfway -> ID switch
            pred_rows.append(f"{frame},{tid},{x},{y},{w},{h},0.9,-1,-1,-1")
        (data_dir / "testseq1.txt").write_text("\n".join(pred_rows) + "\n")

    return gt_dir, trackers_dir


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    root = tmp_path_factory.mktemp("trackeval")
    gt_dir, trackers_dir = _make_benchmark(root)
    return run_trackeval(gt_dir, trackers_dir, ["perfect", "idswitch"])


def test_perfect_tracker_scores_one(benchmark):
    perfect = benchmark["perfect"]["pedestrian"]
    assert perfect["HOTA"]["HOTA"] == pytest.approx(1.0, abs=0.05)
    assert perfect["Identity"]["IDF1"] == pytest.approx(1.0, abs=0.05)
    assert perfect["CLEAR"]["MOTA"] == pytest.approx(1.0, abs=0.05)


def test_id_switch_degrades_idf1(benchmark):
    idswitch = benchmark["idswitch"]["pedestrian"]
    perfect = benchmark["perfect"]["pedestrian"]
    assert idswitch["Identity"]["IDF1"] < perfect["Identity"]["IDF1"] - 0.2
    assert idswitch["HOTA"]["HOTA"] < perfect["HOTA"]["HOTA"]
