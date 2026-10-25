# Final report

Short summary of the system, measured results, and known limits. All numbers
come from real, seeded runs; artifacts not yet run are marked **not run**.
Detailed tables: `docs/RESULTS.md`, `docs/ROBUSTNESS.md`; data sources and
licenses: `docs/DATA.md`; privacy/ethics: `docs/PRIVACY.md`.

## System

Video -> YOLO detection -> ByteTrack tracking -> scene config (stop line,
lanes, signal ROI) -> signal-state heuristic with temporal smoothing -> rule
engine (red-light, wrong-way) -> evidence builder (blurred clip + keyframe +
JSON) -> SQLite -> FastAPI + Streamlit review dashboard. Phase 6 adds an
optional helmet rule (secondary detector on motorcycle crops), an optional
ANPR module, and a robustness evaluation.

Every output is a candidate event for human review, never an automated
penalty. Plates/faces are blurred in stored evidence by default.

## Measured results

Detection (val splits, seed 42; per-class tables in `docs/RESULTS.md`):

| model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| vehicle detector (`weights/detector.pt`, UA-DETRAC val, 500 imgs) | 0.9702 | 0.8407 |
| helmet detector (`weights/helmet.pt`, 372 imgs) | 0.9204 | 0.5864 |
| plate detector (`weights/plate.pt`, 2048 imgs) | 0.9738 | 0.6873 |

Robustness (vehicle detector under deterministic degradations, UA-DETRAC val;
mAP@0.5): night 0.9466, rain 0.9698, glare 0.9694 vs clean 0.9702.

Rule engine (7 synthetic-trajectory scenarios): precision 1.000, recall 1.000,
F1 1.000, 0.0 false alerts/hour. The scenario suite is synthetic by
construction and acts as unit tests.

Real-video event set (3 clips, 20.1 s of UA-DETRAC footage; manifest
`configs/event_set.yaml`): precision 1.000, recall 1.000, F1 1.000,
0.0 false alerts/hour. Breakdown: negative clip (normal traffic) 0 false
alerts; constructed wrong-way clip (reversed playback) 7/7 verified events;
constructed red-light clip (synthetic signal overlay) 2/2 genuine events.

This set caught two real engineering lessons, documented in
`configs/event_set.yaml`: (1) an occlusion-driven bounding-box re-cross
produced a spurious red-light event — fixed with a per-track recross window
(`red_light_recross_s`, default 5 s) that suppresses crossings onto a side
the track already occupied recently; before the fix the set scored
precision 0.800 / F1 0.889. (2) What first looked like a tracker ID-switch
duplicate turned out, on frame-by-frame video review, to be a genuine second
violator following the first — ground truth was corrected, not the metric.
Red-light recall rests on two constructed events; wrong-way recall is a lower
bound (tracker fragmentation). This is a smoke test, not field performance.

Tracking: ByteTrack is integrated and produces stable IDs on the sample clips
used during development; the UA-DETRAC benchmark eval (HOTA/IDF1/ID switches)
is **not run** — dataset access was requested by the repo owner and is still
pending (see `docs/DATA.md`).

System: the pipeline processes the 50-frame synthetic demo clip at ~23 fps on
an Apple-silicon laptop (MPS). No Jetson/ONNX/TensorRT edge test was done
(stretch goal, **not run**).

## Limitations

- Red-light rule depends on per-camera scene calibration and a visible signal
  ROI; views without one have the rule disabled by config.
- Signal state is a color/ROI heuristic, not a learned classifier.
- Event-level metrics on real video rest on 3 short clips (20.1 s) with a
  handful of constructed violations — a smoke test, not field performance.
- Remaining occlusion risk: long gaps still fragment tracks (wrong-way recall
  is a lower bound); track stitching is the natural follow-up.
- Plate blurring uses a heuristic plate band; the ANPR module does not feed
  the blur step.
- Helmet/plate detectors are trained on third-party datasets (licenses in
  `docs/DATA.md`); check regional fit before reuse.

## Reproduction

Fresh clone: `make setup && make demo` (COCO weights only, no datasets), or
`docker build -t tvd . && docker run --rm tvd`. Full pipeline with trained
weights: see README (weights and datasets are gitignored; training notebooks
in `notebooks/`).
