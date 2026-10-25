# Traffic Violation Detection

End-to-end pipeline that detects and tracks vehicles in traffic video, understands
the scene (lanes, stop line, signal state), applies rules to flag candidate
violations, and produces reviewable evidence through a dashboard.

**Scope note:** output is always a *candidate event for human review* — never an
automated penalty. Plates and faces are blurred in stored outputs by default.

MVP violations: red-light running, wrong-way driving.
Extensions: helmet non-use, illegal turn, stopped-vehicle detection, optional ANPR.

## Setup

```bash
make setup        # create .venv and install pinned dependencies
make test         # run the test suite
make lint         # ruff check + format check
```

## Quick start

```bash
make run          # generate a synthetic sample video into data/samples/
```

## Demo (fresh clone, no trained weights)

`make demo` generates the synthetic sample video, runs the full pipeline with
the COCO `yolov8n.pt` weights (auto-downloaded) and all extension modules
disabled (`configs/extensions.demo.yaml`), then runs the rule-engine eval.
It needs no datasets or trained weights:

```bash
make demo
```

To use the trained vehicle/helmet/plate detectors instead, point
`--weights weights/detector.pt` and `configs/extensions.yaml` (Phase 6).

## Demo video (real clip)

With the trained detector weights in place (`weights/detector.pt`), render an
annotated tracking video of the bundled UA-DETRAC sample clip:

```bash
python scripts/run_video.py --video data/samples/detrac_mvi20012.mp4 \
    --weights weights/detector.pt --run-name detrac-demo
# -> results/tracking/detrac-demo/annotated.mp4 (gitignored)
```

The clip shows normal traffic, so the rule engine correctly flags no events;
it demonstrates detection and tracking, not a violation. For flagged-event
evidence clips, run `scripts/run_pipeline.py` on footage containing a
violation (evidence lands in `results/pipeline/<run>/evidence/`).

## Docker

CPU-only image that reproduces the demo (and can run the test suite) with no
datasets or trained weights:

```bash
docker build -t tvd .
docker run --rm tvd                          # synthetic demo + rule eval
docker run --rm tvd python -m pytest -q      # test suite instead
```

The image downloads the COCO `yolov8n.pt` weights on first run; outputs land
under `results/` inside the container.

## Detection (Phase 1)

```bash
# put a YOLO-format dataset in data/ (see docs/DATA.md), then:
python scripts/train_detector.py --train-config configs/train.yaml
python scripts/eval_detector.py --weights weights/detector.pt
# pipeline smoke test (tiny coco8 sample, 1 epoch):
python scripts/train_detector.py --train-config configs/train.smoke.yaml
```

The eval script regenerates `docs/RESULTS.md` (committed) and
`results/detection/<run>/metrics.json` (gitignored).

## Tracking (Phase 2)

```bash
# track a video: writes MOT tracks, trajectories CSV, annotated mp4, summary
python scripts/run_video.py --video <clip.mp4> --weights weights/detector.pt

# tracking metrics (HOTA/IDF1/ID switches/MOTA) against MOT ground truth:
# 1) obtain UA-DETRAC video + annotation XMLs (request form, see docs/DATA.md)
# 2) python scripts/convert_detrac_to_mot.py --annotations-dir <xml dir> \
#      --videos-dir <avi dir> --out-dir data/detrac_mot
# 3) python scripts/eval_tracking.py --video <avi dir>/<seq>.avi \
#      --weights weights/detector.pt --gt-folder data/detrac_mot
```

Note: trackeval 1.3.0 only allows the 'pedestrian' class slot for custom MOT
benchmarks; vehicle GT is mapped to that slot (class id 1).

## Scene calibration + signal state (Phase 3)

```bash
# interactively annotate a camera view (stop line, lanes+directions, signal ROI)
python scripts/annotate_scene.py --camera-id cam-01 \
    --video <clip.mp4> --out configs/scene.cam-01.yaml

# signal state from an ROI (heuristic baseline with majority-vote smoothing)
python -c "from tvd.signal import classify_roi, SignalState; ..."
```

Committed scene configs (`configs/scene.*.yaml`) are covered by
`tests/test_scene_configs.py`: every view must validate (Phase 3 exit).

## Rule engine (Phase 4)

Red-light (stop-line crossing while signal red, with tolerance window) and
wrong-way (sustained opposing motion) rules run on synthetic trajectories:

```bash
python scripts/eval_rules.py   # synthetic scenarios: event-level P/R/F1 + false alerts/hour

# real-video event set (manifest configs/event_set.yaml; build derived clips first):
python scripts/build_event_set.py
python scripts/eval_events.py
```

The real-video set covers a negative clip (normal traffic), a constructed
wrong-way clip (reversed playback), and a constructed red-light clip
(synthetic signal overlay, 2 genuine events). All outputs are candidates for
human review.

## Evidence + dashboard (Phase 5)

```bash
# end-to-end: video -> detect -> track -> signal -> rules -> evidence -> SQLite
python scripts/run_pipeline.py --video <clip.mp4> --scene configs/scene.<cam>.yaml \
    --weights weights/detector.pt

# review API (serves events + blurred evidence clips)
uvicorn tvd.api.main:app --port 8000

# review dashboard
streamlit run src/tvd/ui/dashboard.py -- --db results/events.db
```

Plates/faces are blurred in stored evidence by default (the blur region is a
heuristic plate band). Cameras with no visible signal set `signal_roi: null`,
which disables the red-light rule for that view.

## Repository layout

```
configs/            model, tracker, system, and per-camera scene configs (YAML)
src/tvd/            library code: detect/ track/ scene/ signal/ rules/ evidence/ api/ ui/ utils/
scripts/            train, eval, run_video, sample-data generators
tests/              pytest suite (includes rule tests on synthetic trajectories)
data/               gitignored; see docs/DATA.md for sources and licenses
results/            gitignored run outputs
docs/               data licenses, privacy/ethics notes
```

## Reproducibility

All runs are seeded (see `configs/system.yaml`) and log their config.
Results files are regenerated from real runs only.

## Privacy and ethics

See `docs/DATA.md` for dataset licenses and `docs/` for the privacy note.
Raw footage of people is never committed to this repository.
